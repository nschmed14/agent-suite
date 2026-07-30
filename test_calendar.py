import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.tools.calendar_tool import authenticate_google_calendar, get_events


def main() -> None:
    try:
        service = authenticate_google_calendar()
        events = get_events(service=service, days=3)
        print(f"Found {len(events)} upcoming events")
        for event in events[:3]:
            print(f"- {event.get('summary', '(No title)')} | {event.get('start')} -> {event.get('end')}")
    except Exception as exc:
        print(f"Calendar test failed: {exc}")


if __name__ == "__main__":
    main()
