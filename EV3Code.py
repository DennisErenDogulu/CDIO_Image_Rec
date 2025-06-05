#!/usr/bin/env python3
"""
tcp_server.py  –  EV3dev TCP Server (Python 3.5-kompatibel)

Hver indkommende TCP-forbindelse skal sende én enkelt JSON-linje (`\n`-termineret) 
i ét af to formater:

  A) {"turn": <antal grader>, "drive": <antal cm>}
     → Robotten drejer ‘turn’ grader og kører ‘drive’ cm frem, aktiverer collector, 
       bakker 30 cm, og svarer “OK\n”.

  B) {"path": [[gx1, gy1], [gx2, gy2], …], "heading": "<E|N|S|W>"}
     → Robotten følger et grid <optionelt multi-hop>:
          1. Drej så du peger mod næste grid-celle.
          2. Kør distancen til næste celle (i cm), aktiver collector, bak 30 cm.
          3. Gentag for hver overgang.
       Til sidst svarer den “DONE\n”.

Ved ugyldigt eller ufuldstændigt JSON svarer den “ERROR\n”.
"""

import socket
import threading
import json
import math

from ev3dev2.motor    import OUTPUT_B, OUTPUT_C, OUTPUT_A, MediumMotor
from ev3dev2.motor    import MoveDifferential, SpeedRPM
from ev3dev2.wheel    import EV3Tire
from ev3dev2.display  import Display

# ───────── Hardware-opsætning ─────────
AXLE_TRACK_MM = 104                    # Akselafstand (mm)
mdiff         = MoveDifferential(
    OUTPUT_B, OUTPUT_C, EV3Tire, AXLE_TRACK_MM
)
collector     = MediumMotor(OUTPUT_A)   # Collector-motor
display       = Display()              # EV3-skærm

# Global state
robot_heading_deg = 0.0                # Aktuel robot-heading i grader (0° = +X, CCW positiv)


# ───────── Hjælpefunktioner ─────────

def grid_to_cm(gx, gy):
    """
    Konverter grid-koordinat (gx, gy) til (x_cm, y_cm) – center af grid-celle.
    Hver celle er GRID_SPACING_CM × GRID_SPACING_CM.
    """
    GRID_SPACING_CM = 2
    x_cm = (gx + 0.5) * GRID_SPACING_CM
    y_cm = (gy + 0.5) * GRID_SPACING_CM
    return x_cm, y_cm


def compute_turn_angle(dx, dy, current_heading):
    """
    Beregn den mindste, signerede vinkel (i grader), som robotten skal dreje,
    for at pege i retning af vektoren (dx, dy). 0° svarer til +X-aksen, positiv retning CCW.
    """
    theta_deg = math.degrees(math.atan2(dy, dx))
    delta     = (theta_deg - current_heading + 180.0) % 360.0 - 180.0
    return delta


def drive_and_collect(turn_deg, dist_cm):
    """
    Udfør én “drej og kør”-sekvens:
      1. Drej med turn_deg grader (positiv = CCW).
      2. Kør dist_cm cm fremad.
      3. Aktiver collector-motor i 0.8 s (50% hastighed).
      4. Bak 30 cm (med samme hastighed).
    Skriver status til EV3-skærmen vha. text_pixels().
    """
    # Vis på EV3-skærmen
    display.clear()
    display.text_pixels(
        "TURN {:.1f}\u00B0".format(turn_deg),
        x=10, y=10, text_color='white'
    )
    display.text_pixels(
        "DRIVE {:.1f} cm".format(dist_cm),
        x=10, y=40, text_color='white'
    )
    display.update()

    # 1) Drej
    mdiff.turn_degrees(SpeedRPM(40), turn_deg)

    # 2) Kør frem (cm → mm)
    mdiff.on_for_distance(SpeedRPM(40), dist_cm * 10)

    # 3) Aktiver collector i 0.8 s
    collector.on_for_seconds(50, 0.8)

    # 4) Bak 30 cm (→ 300 mm)
    mdiff.on_for_distance(SpeedRPM(40), -300)

    # Ryd skærmen igen
    display.clear()
    display.update()


# ───────── Client-handler ─────────

def handle_client(conn, addr):
    """
    Kører i en separat tråd for hver ny forbindelse.
    Modtager præcis én \n-termineret JSON-linje og prøver at behandle den
    som enten kort “turn/drive” eller længere “path/heading”.
    Sender tilbage “OK\n” eller “DONE\n” ved succes, ellers “ERROR\n”.
    """
    global robot_heading_deg
    print("[TCP Server] Client connected: {}".format(addr))

    # 1) Læs én linje (slutter ved '\n')
    data_buffer = ""
    while True:
        chunk = conn.recv(1024)
        if not chunk:
            print("[TCP Server] Client {} disconnected før data.".format(addr))
            conn.close()
            return
        data_buffer += chunk.decode("utf-8")
        if "\n" in data_buffer:
            json_str, _ = data_buffer.split("\n", 1)
            break

    # 2) Parse JSON
    try:
        request = json.loads(json_str)

        # 2A) Hvis vi har "turn" + "drive" → kort kommando
        if "turn" in request and "drive" in request:
            turn_val  = float(request["turn"])
            drive_val = float(request["drive"])
            drive_and_collect(turn_val, drive_val)
            conn.sendall(b"OK\n")
            print("[TCP Server] turn/drive udført")
            conn.close()
            return

        # 2B) Ellers hent path + heading
        path    = request.get("path", [])
        heading = request.get("heading", "E").upper()

    except Exception as e:
        print("[TCP Server] JSON parse error: {}".format(e))
        conn.sendall(b"ERROR\n")
        conn.close()
        return

    # 3) Hvis path har færre end 2 punkter, gør ingenting
    if len(path) < 2:
        conn.sendall(b"DONE\n")
        print("[TCP Server] Tomt eller enkelt-punkts path – DONE uden bevægelse.")
        conn.close()
        return

    # 4) Konverter heading-bogstav til grader
    heading_map = {
        "E": 0.0,
        "N": 90.0,
        "W": 180.0,
        "S": -90.0
    }
    robot_heading_deg = heading_map.get(heading, 0.0)
    print("[TCP Server] Received path: {}, initial heading: {}°"
          .format(path, robot_heading_deg))

    # 5) Følg path-parrene
    for i in range(len(path) - 1):
        gx1, gy1 = path[i]
        gx2, gy2 = path[i + 1]

        x1_cm, y1_cm = grid_to_cm(gx1, gy1)
        x2_cm, y2_cm = grid_to_cm(gx2, gy2)

        dx = x2_cm - x1_cm
        dy = y2_cm - y1_cm

        # Drej mod ny vinkel
        delta_deg = compute_turn_angle(dx, dy, robot_heading_deg)
        drive_and_collect(delta_deg, 0)
        robot_heading_deg = (robot_heading_deg + delta_deg) % 360.0

        # Kør frem, aktiver collector, bak 30 cm
        distance_cm = math.hypot(dx, dy)
        drive_and_collect(0, distance_cm)

    # 6) Udført – send DONE tilbage
    conn.sendall(b"DONE\n")
    print("[TCP Server] Path execution complete; sent DONE.")
    conn.close()
    print("[TCP Server] Client {} disconnected.".format(addr))


# ───────── Main-serverloop ─────────

def main():
    HOST = ""      # Lyt på alle interfaces
    PORT = 12345   # Skal matche PC-klientens port

    print("[TCP Server] Starting on port {} …".format(PORT))
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print("[TCP Server] Listening for incoming connections on port {}…".format(PORT))

    try:
        while True:
            conn, addr = server_sock.accept()
            client_thread = threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True
            )
            client_thread.start()
    except KeyboardInterrupt:
        print("\n[TCP Server] Shutting down…")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()