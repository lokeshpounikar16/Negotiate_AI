# NegotiationAI Dashboard

This is the Week 4 frontend scaffold for the negotiation monitoring UI.

## Run locally

```bash
cd dashboard
npm install
npm run dev
```

The dashboard will run on http://localhost:3000.

## Notes

- It reads the backend status from the FastAPI app at http://localhost:8000/health.
- The current starter version uses a demo negotiation feed so the interface is visible even before live API traffic arrives.
- The app is ready to be extended with WebSocket streaming and richer analytics in the next iteration.
