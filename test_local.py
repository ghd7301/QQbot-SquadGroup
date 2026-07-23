import json
import urllib.request


def ask(question: str) -> None:
    data = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8088/ask",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    print(body["answer"])


if __name__ == "__main__":
    ask("HAB 和 FOB 有什么区别？")
