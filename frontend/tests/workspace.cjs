/* Run against Vite with npm run test:ui. Uses synthetic audio and a mock ASR server. */
const { chromium } = require("playwright");
const assert = require("node:assert/strict");

(async () => {
  const browser = await chromium.launch({
    ...(process.env.CHROME_PATH
      ? { executablePath: process.env.CHROME_PATH }
      : {}),
    headless: true,
    args: [
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
    ],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 900 },
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/api/status", (route) =>
      route.fulfill({
        json: {
          status: "ready",
          device: "cpu",
          cleanup_available: true,
          languages: [
            { label: "Auto-detect", value: "auto" },
            { label: "English (en-US)", value: "en-US" },
            { label: "Hindi (hi-IN)", value: "hi-IN" },
          ],
          profiles: ["Lowest latency", "Balanced", "Accurate", "Most accurate"],
          default_profile: "Balanced",
        },
      }),
    );
    let server;
    let starts = 0;
    let chunks = 0;
    await page.routeWebSocket("**/ws/transcribe", (ws) => {
      server = ws;
      ws.onMessage((message) => {
        if (typeof message !== "string") {
          chunks++;
          return;
        }
        const event = JSON.parse(message);
        if (event.type === "start") {
          starts++;
          ws.send(JSON.stringify({ type: "started" }));
        }
        if (event.type === "end") ws.send(JSON.stringify({ type: "ended" }));
      });
    });
    const send = (event) => server.send(JSON.stringify(event));
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    await page.locator("#pedal:enabled").waitFor();
    if (process.env.SCREENSHOT_DIR)
      await page.screenshot({
        path: `${process.env.SCREENSHOT_DIR}/wisper-new-empty.png`,
      });
    await page.locator("#language-button").click();
    await page.locator("#language-search").fill("hindi");
    await page.locator("#language-options button").click();
    assert.equal(
      await page.locator("#language-label").textContent(),
      "Hindi (hi-IN)",
    );
    await page.locator("#language-button").click();
    await page.keyboard.press("Escape");
    assert.equal(
      await page.locator("#language-dialog").evaluate((el) => el.open),
      false,
    );
    await page.locator("input[name=recognition][value=Accurate]").check();
    assert.equal(await page.locator("#profile").inputValue(), "Accurate");
    await page.locator("#pedal").click();
    await page.waitForFunction(
      () => document.body.dataset.phase === "listening",
    );
    assert.equal(await page.locator("#language-button").isDisabled(), true);
    await page.waitForTimeout(200);
    const before = await page.locator("#pedal").boundingBox();
    const raw =
      "I talk about a humanizer model. Never mind, let's talk about the AI detection model.";
    const clean = "Let's talk about the AI detection model.";
    send({ type: "commit", id: 1, raw, cleanup_status: "pending" });
    await page.waitForFunction(() =>
      document
        .querySelector("#cleanup-status")
        .textContent.includes("Polishing"),
    );
    send({
      type: "cleaned",
      id: 1,
      raw,
      cleaned: clean,
      cleanup_status: "applied",
    });
    await page.waitForFunction(
      () =>
        document.querySelector(".sentence")?.textContent ===
        "Let's talk about the AI detection model.",
    );
    await page.locator("#view-original").click();
    assert.equal(await page.locator(".sentence").textContent(), raw);
    await page.locator("#view-polished").click();
    assert.equal(await page.locator(".sentence").textContent(), clean);
    if (process.env.SCREENSHOT_DIR)
      await page.screenshot({
        path: `${process.env.SCREENSHOT_DIR}/wisper-new-writing.png`,
      });
    const paragraph =
      "This is a long dictation. The transcript should scroll inside the writing area while the recording controls stay visible. ".repeat(
        4,
      );
    for (let id = 2; id <= 45; id++)
      send({
        type: "cleaned",
        id,
        raw: paragraph,
        cleaned: paragraph,
        cleanup_status: "applied",
      });
    await page.waitForFunction(
      () => document.querySelectorAll(".sentence").length === 45,
    );
    await page.waitForTimeout(150);
    const after = await page.locator("#pedal").boundingBox();
    assert.ok(
      Math.abs(before.y - after.y) < 1,
      "Recording button moved with transcript growth",
    );
    assert.ok(
      await page
        .locator("#transcript-scroll")
        .evaluate((el) => el.scrollHeight > el.clientHeight),
      "No internal transcript scroll",
    );
    await page.locator("#transcript-scroll").evaluate((el) => {
      el.scrollTop = 0;
    });
    await page.waitForTimeout(50);
    send({
      type: "partial",
      full: "More words",
      live: "More words",
      detected_lang: "",
    });
    await page.waitForTimeout(50);
    assert.equal(
      await page.locator("#transcript-scroll").evaluate((el) => el.scrollTop),
      0,
      "New text interrupted reading",
    );
    await page.locator("#jump-live").click();
    await page.waitForTimeout(50);
    assert.ok(
      await page
        .locator("#transcript-scroll")
        .evaluate((el) => el.scrollTop > 0),
    );
    send({
      type: "cleaned",
      id: 46,
      raw: "Keep my words.",
      cleaned: "Keep my words.",
      cleanup_status: "failed",
    });
    await page.waitForFunction(
      () =>
        document.querySelector("#cleanup-status").dataset.state === "failed",
    );
    for (const [width, height] of [
      [1280, 720],
      [390, 844],
      [320, 568],
    ]) {
      await page.setViewportSize({ width, height });
      await page.waitForTimeout(50);
      const bounds = await page.locator("#pedal").boundingBox();
      assert.ok(
        bounds.y >= 0 && bounds.y + bounds.height <= height,
        `Recording button off screen at ${width}`,
      );
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        `Horizontal overflow at ${width}`,
      );
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollHeight <= innerHeight,
        ),
        `Page scroll at ${width}`,
      );
      if (width === 390 && process.env.SCREENSHOT_DIR)
        await page.screenshot({
          path: `${process.env.SCREENSHOT_DIR}/wisper-new-mobile.png`,
        });
    }
    await page.locator("#pedal").click();
    await page.waitForFunction(() => document.body.dataset.phase === "idle");
    await page.locator("#pedal").click();
    await page.waitForFunction(
      () => document.body.dataset.phase === "listening",
    );
    await page.locator("#pedal").click();
    await page.waitForFunction(() => document.body.dataset.phase === "idle");
    assert.equal(starts, 2);
    assert.ok(chunks > 0, "No microphone PCM delivered");
    assert.deepEqual(errors, []);
    console.log(
      "Passed: long transcript, fixed recorder, mobile widths, language picker, recognition cards, cleanup states, original/polished, reading position, live PCM, and restart.",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
