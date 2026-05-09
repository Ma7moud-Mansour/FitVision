// ─── FitVision API Client ───
const API = {
    BASE: '/api/v1',
    token: localStorage.getItem('fv_token'),
    user: JSON.parse(localStorage.getItem('fv_user') || 'null'),

    headers() {
        const h = { 'Accept': 'application/json' };
        if (this.token) h['Authorization'] = `Bearer ${this.token}`;
        return h;
    },

    async post(path, body, isForm = false) {
        const opts = { method: 'POST', headers: this.headers() };
        if (isForm) {
            opts.body = body; // FormData
        } else {
            opts.headers['Content-Type'] = 'application/json';
            opts.body = JSON.stringify(body);
        }
        const res = await fetch(this.BASE + path, opts);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Request failed' }));
            throw new Error(err.detail || 'Request failed');
        }
        return res.json();
    },

    async get(path) {
        const res = await fetch(this.BASE + path, { headers: this.headers() });
        if (res.status === 401) { this.logout(); return; }
        if (!res.ok) throw new Error('Request failed');
        return res.json();
    },

    async login(username, password) {
        const form = new URLSearchParams();
        form.append('username', username);
        form.append('password', password);
        const res = await fetch(this.BASE + '/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Login failed');
        }
        const data = await res.json();
        this.token = data.access_token;
        localStorage.setItem('fv_token', data.access_token);
        return data;
    },

    async register(email, username, password, fullName) {
        return this.post('/auth/register', { email, username, password, full_name: fullName });
    },

    async uploadVideo(file, exerciseType) {
        const form = new FormData();
        form.append('file', file);
        const res = await fetch(this.BASE + `/upload-video?exercise_type=${exerciseType}`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${this.token}` },
            body: form
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Upload failed');
        }
        return res.json();
    },

    async getSessionStatus(id) { return this.get(`/session/${id}/status`); },
    async getMyWorkouts(skip = 0, limit = 20) { return this.get(`/my-workouts?skip=${skip}&limit=${limit}`); },

    logout() {
        this.token = null;
        this.user = null;
        localStorage.removeItem('fv_token');
        localStorage.removeItem('fv_user');
        window.location.href = '/';
    },

    isLoggedIn() { return !!this.token; },

    requireAuth() {
        if (!this.isLoggedIn()) { window.location.href = '/'; return false; }
        return true;
    }
};
