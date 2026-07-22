function getBackendBaseUrl() {
  const override = window.AGENT_SUITE_BACKEND_PORT || window.__AGENT_SUITE_BACKEND_PORT__;
  const port = override ? String(override) : '8000';
  return `${window.location.protocol}//${window.location.hostname || 'localhost'}:${port}`;
}

class UIOverlay {
  constructor(scene) {
    this.scene = scene;
    this.messages = [
      {
        role: 'assistant',
        text: 'Hello. I can help you coordinate the office, assign tasks, and keep the manager informed.',
      },
    ];

    this.container = document.createElement('div');
    this.container.style.position = 'fixed';
    this.container.style.left = '0';
    this.container.style.top = '0';
    this.container.style.right = '0';
    this.container.style.bottom = '0';
    this.container.style.width = '100vw';
    this.container.style.height = '100vh';
    this.container.style.pointerEvents = 'auto';
    this.container.style.zIndex = '30';
    this.container.style.overflow = 'hidden';
    document.body.appendChild(this.container);

    this.launcher = document.createElement('button');
    this.launcher.textContent = 'Speak to the manager';
    this.launcher.style.position = 'absolute';
    this.launcher.style.left = '50%';
    this.launcher.style.bottom = '24px';
    this.launcher.style.transform = 'translateX(-50%)';
    this.launcher.style.padding = '12px 20px';
    this.launcher.style.borderRadius = '999px';
    this.launcher.style.border = '1px solid rgba(255,255,255,0.45)';
    this.launcher.style.background = '#f7f2e8';
    this.launcher.style.color = '#1f2630';
    this.launcher.style.fontWeight = '700';
    this.launcher.style.cursor = 'pointer';
    this.launcher.style.boxShadow = '0 10px 30px rgba(0,0,0,0.25)';
    this.launcher.style.zIndex = '40';
    this.launcher.onclick = () => this.openPanel();
    this.container.appendChild(this.launcher);

    this.panel = document.createElement('div');
    this.panel.style.position = 'absolute';
    this.panel.style.left = '0';
    this.panel.style.right = '0';
    this.panel.style.top = '0';
    this.panel.style.bottom = '0';
    this.panel.style.display = 'none';
    this.panel.style.pointerEvents = 'auto';
    this.panel.style.padding = '24px';
    this.panel.style.background = 'rgba(247, 242, 232, 0.98)';
    this.panel.style.backdropFilter = 'blur(10px)';
    this.panel.style.overflow = 'hidden';
    this.container.appendChild(this.panel);

    const shell = document.createElement('div');
    shell.style.display = 'flex';
    shell.style.gap = '16px';
    shell.style.height = '100%';
    this.panel.appendChild(shell);

    this.historyPanel = document.createElement('div');
    this.historyPanel.style.flex = '1';
    this.historyPanel.style.background = '#ffffff';
    this.historyPanel.style.borderRadius = '24px';
    this.historyPanel.style.padding = '18px 18px 16px';
    this.historyPanel.style.display = 'flex';
    this.historyPanel.style.flexDirection = 'column';
    this.historyPanel.style.minWidth = '0';
    this.historyPanel.style.boxShadow = '0 18px 45px rgba(0,0,0,0.12)';
    shell.appendChild(this.historyPanel);

    const historyHeader = document.createElement('div');
    historyHeader.innerHTML = '<strong>Conversation box</strong>';
    historyHeader.style.marginBottom = '12px';
    historyHeader.style.color = '#1f2630';
    this.historyPanel.appendChild(historyHeader);

    this.historyEl = document.createElement('div');
    this.historyEl.style.flex = '1';
    this.historyEl.style.overflowY = 'auto';
    this.historyEl.style.display = 'flex';
    this.historyEl.style.flexDirection = 'column';
    this.historyEl.style.gap = '8px';
    this.historyEl.style.paddingRight = '4px';
    this.historyPanel.appendChild(this.historyEl);

    const inputRow = document.createElement('div');
    inputRow.style.display = 'flex';
    inputRow.style.gap = '8px';
    inputRow.style.marginTop = '12px';
    this.historyPanel.appendChild(inputRow);

    this.inputEl = document.createElement('input');
    this.inputEl.type = 'text';
    this.inputEl.placeholder = 'Ask the manager to plan, summarize, or delegate…';
    this.inputEl.style.flex = '1';
    this.inputEl.style.padding = '12px 14px';
    this.inputEl.style.borderRadius = '999px';
    this.inputEl.style.border = '1px solid rgba(31, 38, 48, 0.15)';
    this.inputEl.style.background = '#f7f2e8';
    this.inputEl.style.color = '#1f2630';
    this.inputEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        this.submitTask();
      }
    });
    inputRow.appendChild(this.inputEl);

    this.sendButton = document.createElement('button');
    this.sendButton.textContent = 'Send';
    this.sendButton.style.padding = '10px 14px';
    this.sendButton.style.borderRadius = '999px';
    this.sendButton.style.border = 'none';
    this.sendButton.style.background = '#f7f2e8';
    this.sendButton.style.color = '#1f2630';
    this.sendButton.style.cursor = 'pointer';
    this.sendButton.onclick = () => this.submitTask();
    inputRow.appendChild(this.sendButton);

    this.managerPane = document.createElement('div');
    this.managerPane.style.flex = '0 0 38%';
    this.managerPane.style.background = 'linear-gradient(135deg, rgba(31,38,48,0.96), rgba(66,79,102,0.95))';
    this.managerPane.style.borderRadius = '24px';
    this.managerPane.style.padding = '16px';
    this.managerPane.style.display = 'flex';
    this.managerPane.style.flexDirection = 'column';
    this.managerPane.style.justifyContent = 'space-between';
    this.managerPane.style.color = '#f7f2e8';
    this.managerPane.style.boxShadow = '0 18px 45px rgba(0,0,0,0.16)';
    shell.appendChild(this.managerPane);

    const managerHeader = document.createElement('div');
    managerHeader.innerHTML = '<strong>Manager</strong><br/>Local AI coordinator';
    managerHeader.style.marginBottom = '10px';
    this.managerPane.appendChild(managerHeader);

    this.managerAvatar = document.createElement('div');
    this.managerAvatar.style.flex = '1';
    this.managerAvatar.style.display = 'flex';
    this.managerAvatar.style.alignItems = 'center';
    this.managerAvatar.style.justifyContent = 'center';
    this.managerAvatar.style.borderRadius = '16px';
    this.managerAvatar.style.background = 'linear-gradient(135deg, rgba(245,200,130,0.2), rgba(255,255,255,0.08))';
    this.managerAvatar.style.fontSize = '20px';
    this.managerAvatar.style.fontWeight = '700';
    this.managerAvatar.style.color = '#f7f2e8';
    this.managerAvatar.textContent = 'M';
    this.managerPane.appendChild(this.managerAvatar);

    this.statusEl = document.createElement('div');
    this.statusEl.style.marginTop = '12px';
    this.statusEl.style.fontSize = '13px';
    this.statusEl.style.color = '#d9c8a7';
    this.managerPane.appendChild(this.statusEl);

    const closeButton = document.createElement('button');
    closeButton.textContent = '✕';
    closeButton.setAttribute('aria-label', 'Close manager conversation');
    closeButton.style.position = 'absolute';
    closeButton.style.top = '16px';
    closeButton.style.left = '16px';
    closeButton.style.padding = '8px 10px';
    closeButton.style.borderRadius = '999px';
    closeButton.style.border = '1px solid rgba(31,38,48,0.15)';
    closeButton.style.background = '#ffffff';
    closeButton.style.color = '#1f2630';
    closeButton.style.cursor = 'pointer';
    closeButton.onclick = () => this.closePanel();
    this.panel.appendChild(closeButton);

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.panel.style.display === 'block') {
        this.closePanel();
      }
    });

    this.renderMessages();
    this.setStatus('Waiting for the next request…');
  }

  setStatus(message) {
    this.statusEl.innerHTML = `<strong>Status:</strong> ${message}`;
  }

  openPanel() {
    this.panel.style.display = 'flex';
    this.panel.style.visibility = 'visible';
    this.panel.style.opacity = '1';
    this.launcher.style.display = 'none';
    this.launcher.style.visibility = 'hidden';
    this.launcher.style.opacity = '0';
    this.inputEl.focus();
  }

  closePanel() {
    this.panel.style.display = 'none';
    this.panel.style.visibility = 'hidden';
    this.panel.style.opacity = '0';
    this.launcher.style.display = 'inline-block';
    this.launcher.style.visibility = 'visible';
    this.launcher.style.opacity = '1';
    this.inputEl.value = '';
  }

  renderMessages() {
    this.historyEl.innerHTML = this.messages
      .map((message) => {
        const isAssistant = message.role === 'assistant';
        const bubbleBackground = isAssistant ? '#f4f6fb' : '#f8e4b0';
        const bubbleColor = '#1f2630';
        return `
          <div style="align-self:${isAssistant ? 'flex-start' : 'flex-end'}; max-width:85%; padding:10px 12px; border-radius:12px; background:${bubbleBackground}; color:${bubbleColor}; border:1px solid rgba(31,38,48,0.08);">
            <strong>${isAssistant ? 'Manager' : 'You'}</strong><br/>${message.text}
          </div>
        `;
      })
      .join('');
    this.historyEl.scrollTop = this.historyEl.scrollHeight;
  }

  appendMessage(role, text) {
    this.messages.push({ role, text });
    this.renderMessages();
  }

  setResponse(message, modelStatus = 'unknown') {
    if (!message) {
      return;
    }
    this.appendMessage('assistant', message);
    if (modelStatus === 'fallback') {
      this.setStatus('Using a lightweight local fallback');
    } else {
      this.setStatus('Manager is ready for the next request');
    }
  }

  setAgentsState(agents) {
    if (!agents || !agents.length) {
      return;
    }

    const summary = agents
      .map((agent) => `${agent.name}: ${agent.status} (${agent.task_progress ?? 0}%)`)
      .join(' • ');
    this.setStatus(summary);
  }

  submitTask(task = null) {
    const requestText = (task || this.inputEl.value).trim();
    if (!requestText) {
      return;
    }

    this.appendMessage('user', requestText);
    this.inputEl.value = '';
    this.setStatus('Thinking locally…');

    const backendUrl = `${getBackendBaseUrl()}/assistant`;
    fetch(backendUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request: requestText }),
    })
      .then((response) => response.json())
      .then((payload) => {
        if (payload && payload.response) {
          this.setResponse(payload.response, payload.model_status || 'unknown');
        }
      })
      .catch(() => {
        this.setStatus('Local backend unavailable');
        this.appendMessage('assistant', 'The local assistant could not be reached.');
      });
  }

  runDemoTask() {
    this.submitTask('Morning briefing');
  }
}
