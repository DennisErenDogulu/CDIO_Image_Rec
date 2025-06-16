import socket
import json
import logging
from typing import List, Tuple

logger = logging.getLogger("client_automated")
logger.setLevel(logging.INFO)

def send_path(
    ip: str,
    port: int,
    path: List[Tuple[int, int]],
    ball_cells: List[Tuple[int, int]],
    heading: str
) -> bool:
    """
    Sender en JSON-melding til EV3s TCP-server:
        {
          "path":       [[gx1, gy1], [gx2, gy2], …],
          "ball_cells": [[bx1, by1], [bx2, by2], …],
          "heading":    "<E|N|S|W>"
        }
    Returnerer True, hvis EV3 svarer "DONE\n", ellers False.
    """
    payload = {
        "path":       path,
        "ball_cells": ball_cells,
        "heading":    heading
    }
    message = json.dumps(payload) + "\n"

    try:
        logger.info("Forbinder til EV3 %s:%d…", ip, port)
        with socket.create_connection((ip, port), timeout=5) as s:
            s.sendall(message.encode("utf-8"))
            logger.info("Sendt JSON → %s", message.strip())

            data = s.recv(32).decode("utf-8").strip()
            logger.info("Modtog svar fra EV3: %s", data)
            return data == "DONE"
    except Exception as e:
        logger.error("Fejl ved send_path: %s", e)
        return False