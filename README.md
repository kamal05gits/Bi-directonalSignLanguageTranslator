---
title: AI-POWERED CONTEXT-AWARE BIDIRECTIONAL INDIAN SIGN LANGUAGE TRANSLATOR WITH MULTILINGUAL VOICE ASSISTANCE AND EMERGENCY ALERT SYSTEM
emoji: 🐨
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
short_description: Bidirectional Indian Sign Language translation with voice and emergency assistance
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# AI-POWERED CONTEXT-AWARE BIDIRECTIONAL INDIAN SIGN LANGUAGE TRANSLATOR WITH MULTILINGUAL VOICE ASSISTANCE AND EMERGENCY ALERT SYSTEM

### Submitted by

- **HARIHARAN A** — 2303601710421019
- **JAIMANI R** — 2303601710421020
- **KAMALESH P** — 2303601710421022

In partial fulfillment for the award of the degree of **Bachelor of Engineering**
in **Computer Science and Engineering**

**Government College of Engineering, Bargur – 635104**

**October 2026**

**Head of the Department**

Dr. J. Nafeesa Begum, M.E., Ph.D.

Professor & Head, Department of CSE

Government College of Engineering, Bargur – 635104

**Supervisor**

Mrs. G. Dhivya

Assistant Professor (Temp), Department of CSE

## Sign to Text

Open **Sign to Text** to recognize ISL gestures from the live camera and build a
text transcript. Select English, Tamil, Telugu, Hindi, Kannada, Malayalam,
Bengali, or Urdu under **Voice language**. **Speak to Text** uses browser speech
recognition for the selected language; **Speak Text** translates the transcript
to that language when needed and reads it aloud. The translated transcript is
also shown below the recognized text.

On **Text to Sign**, choose the phrase language, then type or use **Speak phrase**
to enter it. Non-English phrases are translated into English, shown in the
English preview, and used to create the ISL sign sequence. Translation requires
internet access, and microphone input requires browser microphone permission.
Speech input and output depend on browser and operating-system language support.

## Emergency alert webhook

Use the **Emergency Alert** button in the navigation to send an alert to the
configured webhook. The browser asks for confirmation before sending.
Configure `WEBHOOK_URL` and `WEBHOOK_TOKEN` in the ignored `.env` file or
deployment secrets; the webhook must use HTTPS. The token is sent only by the
server in the `Authorization: Bearer` and `X-Webhook-Token` headers and is never
exposed to the browser. If delivery fails, the alert reports the webhook's HTTP
status (without exposing its response body) to help diagnose URL or
authentication configuration.

The server sends a JSON POST containing `event` (`emergency_alert`), `source`
(`Signora`), a `message`, and an ISO-8601 UTC `timestamp`. The webhook service
must return a 2xx response for the alert to be reported as delivered. The URL
remains server-side and is never sent to the browser. Requests are limited to
one attempt per minute per client IP.

Use the **Model** selector to choose a ready model for the live camera. The
choice is saved in this browser.

- **ISL Landmark CNN** is the default and recognizes the 61 classes in
  `models/trained/cnn_model_labels.pkl`.
- **Swatarnik BiGRU** recognizes word gestures from 40 consecutive frames of
  MediaPipe Holistic landmarks. The loader supports the bundled Keras 3 checkpoint
  with the app's TensorFlow 2.13 environment. Its model, class labels, and
  normalization statistics must all load successfully before it appears as a selectable model.
  BiGRU uses an 8 FPS capture cadence and MediaPipe's lightweight pose model;
  actual prediction time depends on the computer's processing speed.
- **Dynamic LSTM** appears when a trained `models/trained/lstm_model.h5` and
  matching labels are available. It recognizes sequences from 30 consecutive
  hand-landmark frames. Collect sequence data, train it with
  `python -m src.models.train_lstm`, then restart the app to load it.

Only models that are ready for live camera inference are listed in the selector.
`/api/models` reports the current load status of those connected models.
Frame predictions are shown only when they exceed `CONFIDENCE_THRESHOLD` and
agree over `PREDICTION_MIN_STABLE_FRAMES` within the
`PREDICTION_SMOOTHING_WINDOW`. A held sign is committed once; show no hand to
re-arm the same sign. An unconfirmed top candidate is shown with its raw model
score, but is not added to the transcript. The landmark CNN classifies static hand poses and does not
translate continuous sign-language sentences by itself.
Emergency Signs predictions also need a raw model score of at least `0.65`
before they can enter the same consensus filter. This reduces uncertain alerts
and sign commits, but cannot correct confident model errors: the bundled
emergency model scored 75.6% on its recorded test set and is experimental, not
a validated emergency safety system.

## Text to Sign

Open **Text to Sign** to enter a phrase and review it as a sequence of animated
ISL sign videos. Available whole-word clips are shown as word signs; words
without a matching clip are finger-spelled with the available letter clips.
Use Play/Pause, previous/next, the speed selector, Loop, or select a step
directly to navigate the sequence.

The project provides only the Sign to Text and Text to Sign user-facing pages.
