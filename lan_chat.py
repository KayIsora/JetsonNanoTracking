import socket
import threading
import argparse
import sys

BUFFER_SIZE = 4096

def recv_loop(sock):
    try:
        while True:
            data = sock.recv(BUFFER_SIZE)
            if not data:
                print("\n[INFO] Kết nối đã đóng.")
                break
            print(f"\n[PEER] {data.decode('utf-8', errors='replace')}")
            print("> ", end="", flush=True)
    except Exception as e:
        print(f"\n[INFO] Nhận dữ liệu kết thúc: {e}")
    finally:
        try:
            sock.close()
        except:
            pass

def send_loop(sock):
    try:
        while True:
            msg = input("> ")
            if msg.strip().lower() in {"quit", "exit"}:
                try:
                    sock.sendall(b"[INFO] Peer da thoat.")
                except:
                    pass
                break
            sock.sendall(msg.encode("utf-8"))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception as e:
        print(f"[ERROR] Loi gui du lieu: {e}")
    finally:
        try:
            sock.close()
        except:
            pass

def run_server(host, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[INFO] Dang lang nghe tai {host}:{port} ...")
    conn, addr = server.accept()
    print(f"[INFO] Da ket noi tu {addr[0]}:{addr[1]}")
    t = threading.Thread(target=recv_loop, args=(conn,), daemon=True)
    t.start()
    send_loop(conn)
    server.close()

def run_client(host, port):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    print(f"[INFO] Da ket noi den {host}:{port}")
    t = threading.Thread(target=recv_loop, args=(client,), daemon=True)
    t.start()
    send_loop(client)

def main():
    parser = argparse.ArgumentParser(description="LAN chat don gian Windows <-> Ubuntu")
    parser.add_argument("--listen", action="store_true", help="Chay o che do server")
    parser.add_argument("--connect", type=str, help="IP server de ket noi")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP bind server")
    parser.add_argument("--port", type=int, default=5000, help="Cong TCP")
    args = parser.parse_args()

    if args.listen:
        run_server(args.host, args.port)
    elif args.connect:
        run_client(args.connect, args.port)
    else:
        print("Dung mot trong hai cach:")
        print("  python lan_chat.py --listen --port 5000")
        print("  python lan_chat.py --connect 192.168.1.50 --port 5000")
        sys.exit(1)
##### python3 lan_chat.py --listen --port 5000
if __name__ == "__main__":################### python lan_chat.py --connect 192.168.1.50 --port 5000
    main()
    ###hi can I push
    # Open Server on Jetson: python3 lan_chat.py --listen --port 5000
    # Connect from Windows: python lan_chat.py --connect 192.168.1.50 --port 5000
    # OKE DONE