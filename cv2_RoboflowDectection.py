#!/usr/bin/env python3
import sys, cv2, os, math, time, random, json, socket, threading, logging, numpy as np, heapq
from queue import Queue
from typing import List, Tuple, Optional
# Roboflow
from roboflow import Roboflow

# (Antag at der findes en fil robot_connection/client_automated.py med send_path‐funktionen)
# send_path(ip: str, port: int, path: List[Tuple[int,int]], heading: str)
from robot_connection.client_automated import send_path

# ───────── Konfiguration ─────────

# Roboflow‐model‐opsætning
RF_API_KEY    = "qJTLU5ku2vpBGQUwjBx2"
RF_WORKSPACE  = "cdio-nczdp"
RF_PROJECT    = "cdio-golfbot2025"
RF_VERSION    = 12

# Spillefladens fysiske størrelse (cm)
REAL_WIDTH_CM  = 180
REAL_HEIGHT_CM = 120

# Grid‐opsætning (cm pr. celle)
GRID_SPACING_CM = 2

# Ignoreret område (bolde heri tælles ikke med i ruteplanen)
IGNORED_AREA = {
    "x_min": 50, "x_max": 100,
    "y_min": 50, "y_max": 100
}

# Startpunkt (cm)
START_POINT_CM = (20.0, 20.0)

# Mulige mål (A og B) i cm‐koordinater (her defineret som en liste af grid‐celler langs kanten)
GOAL_RANGE = {
    # Målet A er ved højre kant (x = REAL_WIDTH_CM), fra y = 56 til 64 cm
    'A': [(REAL_WIDTH_CM, y_cm) for y_cm in range(56, 65)],
    # Målet B kunne ligge langs en anden kant (eksempelvis venstre kant). Udkommenter / tilpas hvis I vil.
    'B': [(x_cm, REAL_HEIGHT_CM) for x_cm in range(80, 91)]
}

# EV3 (TCP)
EV3_IP   = "172.20.10.6"
EV3_PORT = 12345
EV3_HEADING_DEFAULT = "E"  # Standard “facing” retning, hvis I vil ændre det senere

# Logger‐opsætning
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("VisionGrid")


# ───────── Hjælpefunktioner ─────────

def heuristic(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    """
    Manhattan‐afstand (grid), da vi kun kan bevæge os op/ned/venstre/højre.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start: Tuple[int,int], goal: Tuple[int,int], obstacles: set,
          max_gx: int, max_gy: int) -> List[Tuple[int,int]]:
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Rekonstruer sti
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        x, y = current
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (x + dx, y + dy)
            if (0 <= neighbor[0] <= max_gx and
                0 <= neighbor[1] <= max_gy and
                neighbor not in obstacles):

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

    # Ingen sti fundet
    return []


# ───────── RoboFlowGridTest‐klasse ─────────

class RoboFlowGridTest:
    def __init__(self):
        # Init Roboflow‐API og model
        self.rf       = Roboflow(api_key=RF_API_KEY)
        self.project  = self.rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
        self.model    = self.project.version(RF_VERSION).model

        # Farver pr. klasse (udfyldes løbende)
        self.class_colors = {}

        # Video capture (OpenCV)
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.error("Kunne ikke åbne kamera #%d", 1)
            raise SystemExit(1)
        # Bredde/højde, hvis I vil bruge fuld HD. 
        # Bemærk: Roboflow‐model tager 416 × 416 som input, så vi skalerer alligevel.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Queues og threading
        self.frame_queue  = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)

        self.skip_frames = 3
        self.frame_count = 0

        # Homografi‐kalibrering
        self.calibration_points = []  # 4 hjørne‐klik (pixels)
        self.homography_matrix  = None  # px ↔ cm

        self.real_width_cm   = REAL_WIDTH_CM
        self.real_height_cm  = REAL_HEIGHT_CM
        self.grid_spacing_cm = GRID_SPACING_CM

        # Samling af obstacles (grid‐celler)
        self.obstacles = set()

        # Fundne bolde (i cm), som (x_cm, y_cm, label)
        self.ball_positions_cm = []

        # Fast startpunkt (i cm)
        self.start_point_cm = START_POINT_CM

        # Mulige mål‐områder (i cm)
        self.goal_range = GOAL_RANGE.copy()
        self.selected_goal = 'A'    # Standard mål “A”
        self.placing_goal  = False  # Hvis I vil tillade at klikke og ændre mål

        # Ignoreret rektangel i cm
        self.ignored_area = IGNORED_AREA

        # Fuld grid‐rute (list af grid‐celler), beregnet i draw_full_route()
        self.full_grid_path = []

    # ────── Homografi & klik ──────

    def click_to_set_corners(self, event, x, y, flags, param):
        """
        4 klik i ANY‐vindue sætter px‐koordinater for hjørnerne i denne rækkefølge:
         1) Øverste venstre hjørne (0,0 i cm)
         2) Øverste højre hjørne (REAL_WIDTH_CM, 0)
         3) Nederste højre hjørne (REAL_WIDTH_CM, REAL_HEIGHT_CM)
         4) Nederste venstre hjørne (0, REAL_HEIGHT_CM)
        Når 4 punkter er sat, beregner vi homografi (px → cm).
        Derefter sættes et lille center‐område som obstacles (20 × 20 cm),
        og sørger for, at kanten af pladen er “gåbar” (fjern disse fra obstacles).
        Højre‐klik (RBUTTON) toggler obstacles enkeltvis i grid‐celler.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.calibration_points) < 4:
                self.calibration_points.append((x, y))
                logger.info("Corner %d sat: (%d, %d)", len(self.calibration_points), x, y)

            if len(self.calibration_points) == 4:
                # Byg matrices til homografi Baseret på (cm) → (px)
                dst_pts = np.array([
                    [0, 0],
                    [self.real_width_cm, 0],
                    [self.real_width_cm, self.real_height_cm],
                    [0, self.real_height_cm]
                ], dtype="float32")
                src_pts = np.array(self.calibration_points, dtype="float32")
                self.homography_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
                logger.info("✅ Homografi klar.")
                self._mark_center_obstacle()
                self._ensure_outer_edges_walkable()

        elif event == cv2.EVENT_RBUTTONDOWN and self.homography_matrix is not None:
            # Højreklik: toggle én obstacle‐celle der, hvor vi klikkede
            pt_cm = self.pixel_to_cm(x, y)
            if pt_cm is not None:
                gx, gy = self.cm_to_grid_coords(pt_cm[0], pt_cm[1])
                max_gx = self.real_width_cm // self.grid_spacing_cm
                max_gy = self.real_height_cm // self.grid_spacing_cm
                if 0 <= gx <= max_gx and 0 <= gy <= max_gy:
                    if (gx, gy) in self.obstacles:
                        self.obstacles.remove((gx, gy))
                        logger.info("🚧 Fjernede obstacle ved (%d, %d)", gx, gy)
                    else:
                        self.obstacles.add((gx, gy))
                        logger.info("🚧 Tilføjede obstacle ved (%d, %d)", gx, gy)

    def _mark_center_obstacle(self):
        """
        Fyld en 20 cm × 20 cm firkant i midten som blocking obstacles.
        Det vil typisk være det område, hvor robotten selv står.
        """
        cx = self.real_width_cm  / 2
        cy = self.real_height_cm / 2
        half = 10  # cm
        for x_cm in range(int(cx - half), int(cx + half), self.grid_spacing_cm):
            for y_cm in range(int(cy - half), int(cy + half), self.grid_spacing_cm):
                gx, gy = self.cm_to_grid_coords(x_cm, y_cm)
                self.obstacles.add((gx, gy))
        logger.info("Center (%d×%d cm) markeret som obstacle.", 2*half, 2*half)

    def _ensure_outer_edges_walkable(self):
        """
        Sørg for, at alle grid‐celler langs kanten (x=0, x=max, y=0, y=max) IKKE er obstacles.
        Så robotten kan “gå” hele kanten uden at blive blokeret af center‐obstacler.
        """
        max_gx = self.real_width_cm // self.grid_spacing_cm
        max_gy = self.real_height_cm // self.grid_spacing_cm
        # Fjern corners langs top/bund og venstre/højre.
        for gx in range(max_gx + 1):
            self.obstacles.discard((gx, 0))
            self.obstacles.discard((gx, max_gy))
        for gy in range(max_gy + 1):
            self.obstacles.discard((0, gy))
            self.obstacles.discard((max_gx, gy))
        logger.info("✅ Ydre kanter ryddet for obstacles.")

    # ────── Tegn grid, obstacles, mål, start ──────

    def draw_metric_grid(self, frame: np.ndarray) -> np.ndarray:
        """
        Tegn grid‐linjer (vertikale + horisontale hvert GRID_SPACING_CM),
        obstacles (røde cirkler), mål‐områder (grønne cirkler),
        og startpunkt (blå cirkel) oveni frame (RGB‐billede i px).
        Returnerer en kopi med overlay.
        """
        if self.homography_matrix is None:
            return frame

        overlay = frame.copy()
        max_gx = self.real_width_cm // self.grid_spacing_cm
        max_gy = self.real_height_cm // self.grid_spacing_cm

        # Tegn vertikale grid‐linjer
        for x_cm in range(0, self.real_width_cm + 1, self.grid_spacing_cm):
            p1 = np.array([[[x_cm, 0]]], dtype="float32")
            p2 = np.array([[[x_cm, self.real_height_cm]]], dtype="float32")
            p1_px = cv2.perspectiveTransform(p1, self.homography_matrix)[0][0]
            p2_px = cv2.perspectiveTransform(p2, self.homography_matrix)[0][0]
            cv2.line(overlay, tuple(p1_px.astype(int)), tuple(p2_px.astype(int)), (100,100,100), 1)

        # Tegn horisontale grid‐linjer
        for y_cm in range(0, self.real_height_cm + 1, self.grid_spacing_cm):
            p1 = np.array([[[0, y_cm]]], dtype="float32")
            p2 = np.array([[[self.real_width_cm, y_cm]]], dtype="float32")
            p1_px = cv2.perspectiveTransform(p1, self.homography_matrix)[0][0]
            p2_px = cv2.perspectiveTransform(p2, self.homography_matrix)[0][0]
            cv2.line(overlay, tuple(p1_px.astype(int)), tuple(p2_px.astype(int)), (100,100,100), 1)

        # Tegn obstacles (røde cirkler på celler)
        for (gx, gy) in self.obstacles:
            pt_cm = np.array([[[gx * self.grid_spacing_cm, gy * self.grid_spacing_cm]]], dtype="float32")
            pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0]
            cv2.circle(overlay, tuple(pt_px.astype(int)), 6, (0, 0, 255), -1)

        # Tegn mål‐områder (grønne cirkler)
        for label, pts in self.goal_range.items():
            for (x_cm, y_cm) in pts:
                xy_cm = np.array([[[x_cm, y_cm]]], dtype="float32")
                xy_px = cv2.perspectiveTransform(xy_cm, self.homography_matrix)[0][0]
                cv2.circle(overlay, tuple(xy_px.astype(int)), 4, (0, 255, 0), -1)
            # Skriv “Goal A” eller “Goal B” midt i området
            mid_y = (pts[0][1] + pts[-1][1]) / 2
            mid_cm = np.array([[[pts[0][0], mid_y]]], dtype="float32")
            mid_px = cv2.perspectiveTransform(mid_cm, self.homography_matrix)[0][0]
            cv2.putText(overlay,
                        f"Goal {label}",
                        tuple((mid_px + np.array([10, 0])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        # Tegn startpunkt (blå cirkel)
        pt_cm = np.array([[[self.start_point_cm[0], self.start_point_cm[1]]]], dtype="float32")
        pt_px = cv2.perspectiveTransform(pt_cm, self.homography_matrix)[0][0]
        cv2.circle(overlay, tuple(pt_px.astype(int)), 8, (255, 0, 0), -1)
        cv2.putText(overlay, "Start", tuple((pt_px + np.array([10, 0])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        return overlay

    # ────── Pixel ↔ cm ↔ grid‐konvertering ──────

    def pixel_to_cm(self, px: int, py: int) -> Optional[Tuple[float, float]]:
        if self.homography_matrix is None:
            return None
        # Perspektivtransformation (px → cm)
        pt_px = np.array([[[px, py]]], dtype="float32")
        # Beregn inverse homografi
        inv_h = np.linalg.inv(self.homography_matrix)
        real_pt = cv2.perspectiveTransform(pt_px, inv_h)[0][0]
        x_cm, y_cm = real_pt
        # Flip y‐koordinat, fordi vi i pixel har (0,0) øverst til venstre,
        # men i cm‐koordinater har vi (0,0) ved nederste venstre.
        y_cm_flipped = self.real_height_cm - y_cm
        return (x_cm, y_cm_flipped)

    def cm_to_grid_coords(self, x_cm: float, y_cm: float) -> Tuple[int,int]:
        """
        Konverter (x_cm, y_cm) til (gx, gy) som heltal (grid‐celler). 
        Hver celle er GRID_SPACING_CM × GRID_SPACING_CM cm.
        """
        gx = int(x_cm // self.grid_spacing_cm)
        gy = int(y_cm // self.grid_spacing_cm)
        return (gx, gy)

    # ────── Tegn fuld rute på billedet ──────

    def draw_full_route(self, frame: np.ndarray, ball_positions: List[Tuple[float,float,str]]) -> np.ndarray:
        """
        Tegn ruten:
         1) Beregn “route” i cm (start → hver non-orange bold → orange bold → mål).
         2) For hvert par i denne cm‐rute, kør A* på grid → fuld liste af grid‐celler (self.full_grid_path).
         3) Tegn streger (gule) langs alle grid‐celler i sellest grid‐rute.
         4) Skriv “Total Path: X cm to Goal Y”.
        Returnerer overlayet.
        """
        if self.homography_matrix is None:
            return frame

        # Opdel bolde i non-orange og orange (case-insensitive)
        non_orange_balls = [b for b in ball_positions if b[2].strip().lower() != "orangetabletennisballs"]
        orange_balls     = [b for b in ball_positions if b[2].strip().lower() == "orangetabletennisballs"]

        # Vælg én orange bold, hvis flere
        orange_ball = orange_balls[0] if orange_balls else None

        # Byg ruten i cm‐koordinater
        route_cm = []
        route_cm.append(self.start_point_cm)
        current = self.start_point_cm

        # Tag non-orange bolde i rækkefølge efter nærmeste-manhattan‐heuristik
        remaining = non_orange_balls.copy()
        while remaining:
            next_ball = min(
                remaining,
                key=lambda b: heuristic(self.cm_to_grid_coords(*current), self.cm_to_grid_coords(b[0], b[1]))
            )
            route_cm.append((next_ball[0], next_ball[1]))
            remaining.remove(next_ball)
            current = (next_ball[0], next_ball[1])

        # Tilføj orange bold sidst (hvis der er en)
        if orange_ball is not None:
            route_cm.append((orange_ball[0], orange_ball[1]))
            current = (orange_ball[0], orange_ball[1])

        # Vælg nærmeste mål‐celle i selected_goal
        goal_candidates = self.goal_range.get(self.selected_goal, [])
        if not goal_candidates:
            logger.warning("No goal candidates for %s", self.selected_goal)
            return frame

        # Find den mål‐celle, der er tættest til current
        best_goal_cm = min(
            goal_candidates,
            key=lambda g: heuristic(self.cm_to_grid_coords(*current), self.cm_to_grid_coords(g[0], g[1]))
        )
        route_cm.append(best_goal_cm)

        # Konverter “route_cm” til en fuld liste af grid‐celler vha. A*
        self.full_grid_path = []
        max_gx = self.real_width_cm // self.grid_spacing_cm
        max_gy = self.real_height_cm // self.grid_spacing_cm

        for i in range(len(route_cm) - 1):
            start_cell = self.cm_to_grid_coords(*route_cm[i])
            end_cell   = self.cm_to_grid_coords(*route_cm[i + 1])
            segment = astar(start_cell, end_cell, self.obstacles, max_gx, max_gy)

            # Undgå duplikation, hvis segment[0] = sidste celle i self.full_grid_path
            if self.full_grid_path and segment and segment[0] == self.full_grid_path[-1]:
                segment = segment[1:]

            self.full_grid_path.extend(segment)

        # Tegn hele mall grid‐ruten som gul streg
        overlay = frame.copy()
        path_color = (0, 255, 255)
        total_cm   = 0

        for j in range(len(self.full_grid_path) - 1):
            gx1, gy1 = self.full_grid_path[j]
            gx2, gy2 = self.full_grid_path[j + 1]

            # Konvertér grid‐celler til cm‐center
            pt1_cm = np.array([[[gx1 * self.grid_spacing_cm, gy1 * self.grid_spacing_cm]]], dtype="float32")
            pt2_cm = np.array([[[gx2 * self.grid_spacing_cm, gy2 * self.grid_spacing_cm]]], dtype="float32")
            pt1_px = cv2.perspectiveTransform(pt1_cm, self.homography_matrix)[0][0]
            pt2_px = cv2.perspectiveTransform(pt2_cm, self.homography_matrix)[0][0]

            pt1 = tuple(pt1_px.astype(int))
            pt2 = tuple(pt2_px.astype(int))
            cv2.line(overlay, pt1, pt2, path_color, 3)
            total_cm += self.grid_spacing_cm

        # Skriv total‐afstand og valgt mål
        text = f"Total Path: {total_cm:.0f} cm to Goal {self.selected_goal}"
        cv2.putText(overlay, text,
                    (10, overlay.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, path_color, 2)

        return overlay

    # ────── Tråd: Capture frames ──────

    def capture_frames(self):
        """
        Læser hver frame fra kameraet og putter den i frame_queue,
        men kun hver self.skip_frames'te frame.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Kameraet sendte ingen frame.")
                break
            self.frame_count += 1
            if self.frame_count % self.skip_frames == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

    # ────── Tråd: Process frames ──────

    def process_frames(self):
        """
        Henter en frame fra frame_queue, kører Roboflow‐detektion,
        tegner bounding‐boxes og gemmer cm‐koordinater for bolde.
        Tegner også grid & rute overlay og pusher i output_queue.
        """
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                original = frame.copy()
                self.ball_positions_cm = []

                # Resize til 416×416 før Roboflow
                small = cv2.resize(frame, (416, 416))
                preds = self.model.predict(small, confidence=30, overlap=20).json()

                # Skaleringsfaktorer: px_small → px_original
                scale_x = frame.shape[1] / 416
                scale_y = frame.shape[0] / 416

                # Gå gennem predictions
                for p in preds.get('predictions', []):
                    x_px = int(p['x'] * scale_x)
                    y_px = int(p['y'] * scale_y)
                    w_px = int(p['width'] * scale_x)
                    h_px = int(p['height'] * scale_y)
                    label = p['class']

                    # Vælg farve for denne klasse (danner farve, hvis ny)
                    color = self.class_colors.setdefault(label,
                        (random.randint(0,255), random.randint(0,255), random.randint(0,255)))

                    # Tegn bounding box + label på original‐billedet
                    cv2.rectangle(original,
                                  (x_px - w_px//2, y_px - h_px//2),
                                  (x_px + w_px//2, y_px + h_px//2),
                                  color, 2)
                    cv2.putText(original, label,
                                (x_px - w_px//2, y_px - h_px//2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Konverter ballens midtpunkt til cm‐koordinater
                    cm_coords = self.pixel_to_cm(x_px, y_px)
                    if cm_coords is not None:
                        cx, cy = cm_coords
                        # Tjek ignorér område
                        if not (IGNORED_AREA["x_min"] <= cx <= IGNORED_AREA["x_max"] and
                                IGNORED_AREA["y_min"] <= cy <= IGNORED_AREA["y_max"]):
                            self.ball_positions_cm.append((cx, cy, label))

                # Tegn grid & obstacles & start & mål
                frame_with_grid = self.draw_metric_grid(original)

                # Tegn rute fra start → bolde → mål
                frame_with_route = self.draw_full_route(frame_with_grid, self.ball_positions_cm)

                # Put resultat i output_queue (til display)
                if not self.output_queue.full():
                    self.output_queue.put(frame_with_route)

    # ────── Tråd: Display frames & brugerinput ──────

    def display_frames(self):
        """
        Viser “Live Object Detection” vindue, håndterer klik (til kalibrering
        og obstacle‐toggle) samt tastetryk:
         • 'q': Quit
         • '1': Vælg målet = 'A'
         • '2': Vælg målet = 'B'
         • 's': Send den fulde grid‐rute til EV3 (hvis beregnet)
        """
        cv2.namedWindow("Live Object Detection")
        # Sæt mus‐callback til kalibrering / toggling
        cv2.setMouseCallback("Live Object Detection", self.click_to_set_corners)

        while True:
            if not self.output_queue.empty():
                frame = self.output_queue.get()
                cv2.imshow("Live Object Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.selected_goal = 'A'
                logger.info("✅ Selected Goal A")
            elif key == ord('2'):
                self.selected_goal = 'B'
                logger.info("✅ Selected Goal B")
            elif key == ord('s'):
                # Send path til EV3, hvis vi har en rute
                if self.full_grid_path:
                    # heading: kan tilpasses efter hvordan I vil dreje EV3 til startretning
                    heading = EV3_HEADING_DEFAULT
                    send_path(EV3_IP, EV3_PORT, self.full_grid_path, heading)
                    logger.info("📨 Path sendt: %s", self.full_grid_path)
                else:
                    logger.warning("⚠️ Ingen rute beregnet endnu til at sende.")

    # ────── Public metode: start tråde ──────

    def run(self):
        """
        Starter capture‐, process‐ og display‐tråde. 
        """
        cap_thread  = threading.Thread(target=self.capture_frames, daemon=True)
        proc_thread = threading.Thread(target=self.process_frames, daemon=True)
        disp_thread = threading.Thread(target=self.display_frames, daemon=True)

        cap_thread.start()
        proc_thread.start()
        disp_thread.start()

        cap_thread.join()
        proc_thread.join()
        # display‐tråden afbrydes, når 'q' trykkes, så man genoptager flowet her
        disp_thread.join()

        # Ryd op
        self.cap.release()
        cv2.destroyAllWindows()


# ───────── Main ─────────

if __name__ == "__main__":
    # Sørg for at fange evt. camera_index som argument
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tester = RoboFlowGridTest()
    tester.run()
