# Wisper API — developer quick start

Wisper is a streaming WebSocket API, not a webhook. Your application opens a connection, sends microphone audio, and receives transcript events in real time.

## URLs

```text
Status:    http://32.198.66.247:8000/api/status
WebSocket: ws://32.198.66.247:8000/ws/transcribe
Swagger:   http://32.198.66.247:8000/docs#/
```

Check that the speech model is ready before connecting:

```bash
curl --fail --silent --show-error \
  http://32.198.66.247:8000/api/status
```

Wait until the returned `status` is `ready`.

## JavaScript example

```js
const ws = new WebSocket("ws://32.198.66.247:8000/ws/transcribe");

ws.addEventListener("message", ({ data }) => {
  const event = JSON.parse(data);
  console.log(event.type, event);

  if (event.type === "ready" && event.status === "ready") {
    ws.send(JSON.stringify({
      type: "start",
      language: "auto",
      profile: "Balanced",
      cleanup: true,
    }));
  }

  if (event.type === "partial") {
    // Display event.full or event.live.
  }

  if (event.type === "commit" || event.type === "polished") {
    // Save the committed transcript text.
  }
});

// Send each captured audio chunk as an ArrayBuffer containing raw PCM16 data.
function sendAudio(pcm16ArrayBuffer) {
  if (ws.readyState === WebSocket.OPEN) ws.send(pcm16ArrayBuffer);
}

// Finish the current recording and wait for the `ended` event.
function stopRecording() {
  ws.send(JSON.stringify({ type: "end" }));
}
```

## Audio format

- Mono
- 16,000 Hz sample rate
- Signed 16-bit PCM, little-endian
- Sent as binary WebSocket frames—not JSON or base64

## Message flow

```text
Server -> ready
Client -> { "type": "start", ... }
Server -> started
Client -> binary PCM16 frames
Server -> partial events
Client -> { "type": "end" }
Server -> commit / polished events
Server -> ended
```

Use `{ "type": "commit" }` during a recording to commit the current utterance without ending the session.

## Important notes

- Only one active recording is supported per server process. A second client receives an error and should retry later.
- The current endpoint uses plain HTTP and `ws://`. A website loaded over HTTPS will normally block that connection as mixed content. Put the API behind HTTPS and use `wss://` for production browser integrations.
- The public endpoint currently has no authentication. Add authentication and network restrictions before handling sensitive or production audio.
- Preserve `warning` and `error` messages in client logs; they contain retry guidance.
