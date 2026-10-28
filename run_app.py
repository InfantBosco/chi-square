import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.2)
    print("Opening browser at http://127.0.0.1:8000 ...")
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    print("Starting The Retail Intelligence Ledger server on http://127.0.0.1:8000 ...")
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=False)
