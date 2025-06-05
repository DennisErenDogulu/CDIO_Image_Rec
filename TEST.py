#!/usr/bin/env python3
import argparse, json, socket, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip",      default="172.20.10.6")
    ap.add_argument("--port",    type=int, default=12345)
    ap.add_argument("--turn",    type=float, required=True)
    ap.add_argument("--drive",   type=float, required=True)
    ap.add_argument("--timeout", type=int,   default=10)
    args = ap.parse_args()

    payload = {"turn": args.turn, "drive": args.drive}
    try:
        with socket.create_connection((args.ip, args.port),
                                      timeout=args.timeout) as s:
            s.sendall((json.dumps(payload) + "\n").encode())
            resp = s.recv(32).decode().strip()
            print("Svar fra EV3:", resp)
    except (socket.timeout, ConnectionError) as e:
        sys.exit("TCP-fejl: {}".format(e))

if __name__ == "__main__":
    main()
