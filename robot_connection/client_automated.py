import socket
import json
import logging
from typing import List, Tuple

logger = logging.getLogger("client_automated")
logger.setLevel(logging.INFO)

def send_path(
    ip: str,
    port: int,
    path: List[dict],            # <─ NOW list of dicts, not tuples
    ball_cells: List[Tuple[int, int]],
    heading: str
) -> bool:
    """
    Sends     { "path":[{"x":gx,"y":gy,"rev":false}, …],
                "ball_cells":[[bx,by], …],
                "heading":"E" }
    Waits for "DONE\n" from the EV3.
    """
    payload = {
        "path":       path,
        "ball_cells": ball_cells,
        "heading":    heading
    }
    message = json.dumps(payload) + "\n"

    try:
        logger.info("Connecting to EV3 %s:%d…", ip, port)
        with socket.create_connection((ip, port), timeout=5) as s:
            s.sendall(message.encode("utf-8"))
            logger.info("JSON sent → %s", message.strip())

            data = s.recv(32).decode("utf-8").strip()
            logger.info("Reply from EV3: %s", data)
            return data == "DONE"
    except Exception as e:
        logger.error("send_path error: %s", e)
        return False

