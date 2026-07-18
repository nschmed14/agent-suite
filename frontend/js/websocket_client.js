class WebSocketClient {
  constructor(url, onMessage) {
    this.url = url;
    this.onMessage = onMessage;
    this.socket = null;
    this.reconnectTimer = null;
    this.reconnectAttempts = 0;
    this.connected = false;
  }

  connect() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      return;
    }

    const socket = new WebSocket(this.url);
    this.socket = socket;

    socket.addEventListener('open', () => {
      this.connected = true;
      this.reconnectAttempts = 0;
      this.onMessage({ type: 'connection', connected: true });
      socket.send(JSON.stringify({ type: 'ping' }));
    });

    socket.addEventListener('message', (event) => {
      try {
        const payload = JSON.parse(event.data);
        this.onMessage(payload);
      } catch (error) {
        console.warn('Unable to parse websocket payload', error);
      }
    });

    socket.addEventListener('close', () => {
      this.connected = false;
      this.onMessage({ type: 'connection', connected: false });
      this.scheduleReconnect();
    });

    socket.addEventListener('error', () => {
      this.onMessage({ type: 'connection', connected: false });
    });
  }

  scheduleReconnect() {
    if (this.reconnectAttempts >= 8) {
      return;
    }
    clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts += 1;
      this.connect();
    }, 1000 * Math.min(this.reconnectAttempts + 1, 5));
  }

  send(payload) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload));
    }
  }

  disconnect() {
    clearTimeout(this.reconnectTimer);
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}
