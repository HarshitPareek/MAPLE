// Centralized API wrapper — attaches JWT, handles errors
const API = {
  _token: () => localStorage.getItem('cg_token'),

  _headers(extra = {}) {
    const h = { 'Content-Type': 'application/json', ...extra };
    const t = this._token();
    if (t) h['Authorization'] = `Bearer ${t}`;
    return h;
  },

  async _handle(res) {
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      // Expired or invalid JWT — clear local auth state so the UI updates
      if (res.status === 401 || res.status === 422) {
        localStorage.removeItem('cg_token');
        localStorage.removeItem('cg_user');
      }
      const err = new Error(data.error || data.msg || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return data;
  },

  get(url)         { return fetch(url, { headers: this._headers() }).then(r => this._handle(r)); },
  post(url, body)  { return fetch(url, { method: 'POST',   headers: this._headers(), body: JSON.stringify(body) }).then(r => this._handle(r)); },
  delete(url)      { return fetch(url, { method: 'DELETE', headers: this._headers() }).then(r => this._handle(r)); },
};

// Toast notifications
const Toast = {
  show(msg, type = '') {
    const c = document.getElementById('toast-container');
    if (!c) return;
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    c.appendChild(el);
    setTimeout(() => el.remove(), 3200);
  },
  success(msg) { this.show(msg, 'success'); },
  error(msg)   { this.show(msg, 'error'); },
};
