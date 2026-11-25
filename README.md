# ⚡ LIGHT-LPCA Turbo

**Clone, Paste Your Free API Key, Give a Task & Go to Sleep.**

An ultra-fast, local-first autonomous AI worker & clean Groq-powered web chat built on top of the LIGHT LPCA architecture.

---

## 😴 30-Second Terminal Sleep Mode (Copy & Run)

1. **Clone this repository:**
   ```bash
   git clone https://github.com/harshitpratap1113-glitch/light-lpca-turbo.git
   cd light-lpca-turbo
   pip install -r requirements.txt
   ```
2. **Give a task and go to sleep:**
   ```bash
   python sleep.py "Research top 10 AI agent frameworks in 2026 and build a complete comparison + roadmap"
   ```
   *(First time it asks for your free `GROQ_API_KEY` from [console.groq.com](https://console.groq.com) and saves it locally to `~/.light_lpca_turbo/.env` with `0600` permissions — never inside this repo.)*

3. **Wake up to your finished `MORNING_BRIEFING.md`:**
   - Every step is checkpointed to disk automatically (`~/.light_lpca_turbo/sleep_runs/<run_id>.json`).
   - If your laptop restarts or internet drops, just resume:
     ```bash
     python sleep.py --resume <run_id>
     ```

---

## 🌐 Clean Web Chat UI (Local or Vercel)

Run locally:
```bash
python -m lpca_turbo.cli serve --port 8787
```
Or try the live secured web chat: **https://light-lpca-turbo.vercel.app**

---

## 🛡️ Security
- No API keys are stored in this repository.
- Rate-limited chat endpoint + strict CSP/XSS security headers (`lpca_turbo/security.py`).
- MIT Licensed ([LICENSE](LICENSE)).
