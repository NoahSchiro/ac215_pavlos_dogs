# Race server

A small FastAPI server that runs **two-player races** in the Gaze Maze. Both players
get the same maze; each sees their opponent as a faded "ghost" and the first to reach
the cheese wins.

> **Why this exists.** Our gaze model runs *in the browser* (privacy: webcam frames never
> leave the user's machine; latency: no network round trip per prediction), so a
> prediction API would add nothing. With our TF's approval, this backend is our
> team-specific replacement for the MS3 "working prediction API" requirement, so it is
> meant to be deployed, documented, and tested like one.

## What the server is responsible for

| Job | Where |
| --- | --- |
| Generate a random perfect maze (one path between any two cells) | `app/maze.py` |
| Rooms with a 4-letter join code, max 2 players, race starts when player 2 joins | `app/game.py` |
| **Validate every move** (walls, edges, rate limit, race not started / already over) | `Room.move` in `app/game.py` |
| Decide the winner (first player on the cheese) | `Room.move` |
| Send both players the same state over a WebSocket | `app/main.py` |
| Structured JSON logs for monitoring | `log_event` in `app/main.py` |

The browser only *asks* to move. The server is the referee, so a modified client can't
walk through walls or teleport to the cheese.

## API

### HTTP

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ok", "rooms": 0}` |
| `POST` | `/rooms` | `201 {"code": "ABCD"}`: creates a room |
| `GET` | `/rooms/{code}` | `{"code", "players", "started", "winner"}`, or `404` |

FastAPI also serves interactive docs at `/docs`.

### WebSocket: `/rooms/{code}/ws`

Server to client:

```jsonc
{"type": "welcome", "player": 1, "maze": {"rows": 15, "cols": 15, "walls": [...], "start": {...}, "cheese": {...}}}
{"type": "state", "started": true, "players": {"1": {"row": 0, "col": 1}, "2": {"row": 0, "col": 0}}, "winner": null}
{"type": "rejected", "reason": "wall"}   // also: too_fast, not_started, finished, invalid_direction, invalid_json, unknown_message
{"type": "opponent_left"}
{"type": "error", "reason": "room_not_found"}   // then closes with code 4404 (room_full: 4409)
```

Client to server:

```json
{"type": "move", "direction": "up"}
```

`direction` is one of `up`, `down`, `left`, `right`. A rejected move doesn't disconnect
you; you just don't move.

### Try it by hand

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/rooms            # -> {"code":"FJTD"}
curl http://localhost:8000/rooms/FJTD
```

Or open http://localhost:8000/docs to try every endpoint from the browser.

## Running locally

```bash
cd race_server
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

The UI looks for the backend at `http://localhost:8000` by default. To point it somewhere
else, build the UI with `VITE_RACE_SERVER_URL=https://<your-backend-url>`.

From the repo root, `docker compose up --build race-server ui` starts both (server on port
8000, UI on 8080).

### Configuration

Defaults live in `app/config.py` (the server's counterpart to `ui/src/config.ts`). Each one can be
overridden with an environment variable of the same name:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAZE_ROWS`, `MAZE_COLS` | `15` | Maze size |
| `MIN_MOVE_INTERVAL_MS` | `100` | Fastest a player may move (anti-cheat rate limit) |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `PORT` | `8080` | Port inside the container (Cloud Run sets this) |

## Tests

```bash
uv run pytest        # 18 tests: maze generation, game rules, HTTP + WebSocket API
uv run ruff check .
```

The API tests play a full two-player race over real WebSockets (FastAPI's `TestClient`),
walking the BFS shortest path to the cheese. CI runs all of this on every PR (`race-server` job).

## Deploying to Cloud Run

```bash
gcloud run deploy race-server \
  --source race_server \
  --region us-east1 \
  --allow-unauthenticated \
  --max-instances 1 \
  --timeout 3600
```

- **`--max-instances 1`**: rooms live in this process's memory. With two instances, the
  two players could land on different machines and never see each other. Fine for a class
  demo; scaling out would mean moving rooms to Redis.
- **`--timeout 3600`**: Cloud Run closes requests (including WebSockets) after the timeout;
  an hour covers any race.

Then rebuild the UI with `VITE_RACE_SERVER_URL` set to the printed service URL, and set
`ALLOWED_ORIGINS` to the UI's URL.

## Monitoring

Every important event is logged as one line of JSON to stdout, which Cloud Run forwards
to Cloud Logging:

```json
{"severity": "INFO", "event": "race_finished", "room": "FJTD", "winner": 1, "seconds": 41.3}
```

Events: `room_created`, `player_joined`, `race_started`, `race_finished`, `player_left`
(includes how many moves were rejected). In Logs Explorer:

```
resource.type="cloud_run_revision"
resource.labels.service_name="race-server"
jsonPayload.event="race_finished"
```
