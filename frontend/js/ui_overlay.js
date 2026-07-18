class UIOverlay {
  constructor(scene) {
    this.scene = scene;
    this.container = document.createElement('div');
    this.container.style.position = 'absolute';
    this.container.style.right = '16px';
    this.container.style.top = '16px';
    this.container.style.width = '320px';
    this.container.style.maxHeight = '70vh';
    this.container.style.overflowY = 'auto';
    this.container.style.background = 'rgba(43, 45, 49, 0.96)';
    this.container.style.border = '2px solid rgba(245, 200, 130, 0.5)';
    this.container.style.borderRadius = '10px';
    this.container.style.padding = '14px';
    this.container.style.color = '#f0e6d3';
    this.container.style.zIndex = '20';
    this.container.style.boxShadow = '0 6px 24px rgba(0,0,0,0.35)';
    document.body.appendChild(this.container);

    this.statusEl = document.createElement('div');
    this.statusEl.innerHTML = '<strong>Status:</strong> Starting…';
    this.container.appendChild(this.statusEl);

    this.agentListEl = document.createElement('div');
    this.agentListEl.style.marginTop = '10px';
    this.agentListEl.style.fontSize = '13px';
    this.agentListEl.style.lineHeight = '1.4';
    this.container.appendChild(this.agentListEl);

    this.agentSelect = document.createElement('select');
    this.agentSelect.innerHTML = `
      <option value="manager">Manager</option>
      <option value="financial">Financial</option>
      <option value="vault">Work Vault</option>
      <option value="assistant">Personal Assistant</option>
      <option value="researcher">Researcher</option>
    `;
    this.agentSelect.style.marginTop = '10px';
    this.agentSelect.style.width = '100%';
    this.agentSelect.style.padding = '6px';
    this.container.appendChild(this.agentSelect);

    this.taskInput = document.createElement('input');
    this.taskInput.type = 'text';
    this.taskInput.placeholder = 'Task for the office';
    this.taskInput.style.width = '100%';
    this.taskInput.style.marginTop = '8px';
    this.taskInput.style.padding = '6px';
    this.container.appendChild(this.taskInput);

    this.button = document.createElement('button');
    this.button.textContent = 'Send task';
    this.button.style.marginTop = '8px';
    this.button.style.padding = '6px 8px';
    this.button.onclick = () => this.submitTask();
    this.container.appendChild(this.button);

    this.setStatus('Waiting for office state…');
    this.agentListEl.innerHTML = '<div style="margin-top:8px"><strong>Office Panel</strong><br/>Agent state will appear here.</div>';
  }

  setStatus(message) {
    this.statusEl.innerHTML = `<strong>Status:</strong> ${message}`;
  }

  setAgentsState(agents) {
    if (!agents || !agents.length) {
      this.agentListEl.innerHTML = '<em>No agent data yet.</em>';
      return;
    }

    const rows = agents.map((agent) => {
      const progress = agent.task_progress ?? 0;
      const thought = agent.thoughts ? agent.thoughts : 'Listening locally';
      return `<div style="margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid rgba(245,200,130,0.2)"><strong>${agent.name}</strong><br/>Status: ${agent.status}<br/>Progress: ${progress}%<br/>Thought: ${thought}</div>`;
    });

    this.agentListEl.innerHTML = rows.join('');
  }

  submitTask() {
    const task = this.taskInput.value.trim();
    if (!task) {
      return;
    }

    const agentId = this.agentSelect.value;
    const backendUrl = `${window.location.protocol}//${window.location.hostname}:8000/task/${agentId}`;

    this.setStatus(`Sending task to ${agentId}…`);
    fetch(backendUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task }),
    })
      .then(() => {
        this.setStatus(`Task sent to ${agentId}`);
        this.taskInput.value = '';
      })
      .catch(() => {
        this.setStatus('Local backend unavailable');
      });
  }
}
