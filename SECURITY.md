# Security Policy

## Notice

Vaani Sahayak is an **educational and demonstration project**. It is not designed or audited for production-grade security. Use at your own discretion.

---

## Known Considerations

| Area | Detail |
|------|--------|
| **API Tokens** | HuggingFace tokens and EI credentials are stored in `.env`. Never commit this file. |
| **CORS** | Default configuration allows `localhost` origins only. Restrict `CORS_ORIGINS` in production. |
| **SSL Verification** | `EI_VERIFY_SSL` can be set to `false` for development with self-signed certificates. Always enable in production. |
| **Model Servers** | `server_param1.py` and `server_tts.py` bind to `0.0.0.0` by default. Restrict to `127.0.0.1` if not serving external clients. |
| **User Input** | User queries are passed directly to the LLM context. Consider input sanitization for production deployments. |

---

## User Responsibilities

If you deploy Vaani Sahayak beyond local development, you are responsible for:

- **Authentication & authorization** — the API has no built-in auth
- **Network controls** — firewall rules, VPN, or reverse proxy with TLS
- **Secrets management** — use a vault or secrets manager instead of `.env` files
- **Monitoring** — log access and anomalies
- **Compliance** — ensure your deployment meets applicable data protection regulations

---

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email **security@cld2labs.com** with details
3. Include steps to reproduce and potential impact
4. We will acknowledge receipt within 48 hours

Thank you for helping keep Vaani Sahayak safe.
