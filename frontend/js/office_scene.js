class OfficeViewer {
  constructor(rootElement) {
    this.rootElement = rootElement;
    this.agentElements = new Map();
    this.state = { agents: [] };
    this.renderShell();
    const backendPort = window.location.port === '8080' ? '8010' : '8000';
    this.wsClient = new WebSocketClient(
      `ws://${window.location.hostname || 'localhost'}:${backendPort}/office`,
      (payload) => this.handleSocketMessage(payload)
    );
    this.wsClient.connect();
    this.uiOverlay = new UIOverlay(this);
    this.uiOverlay.setAgentsState(this.state.agents);
  }

  renderShell() {
    this.rootElement.innerHTML = '';
    this.rootElement.style.position = 'relative';
    this.rootElement.style.width = '100vw';
    this.rootElement.style.height = '100vh';
    this.rootElement.style.background = '#2b2d31';
    this.rootElement.style.overflow = 'hidden';

    const room = document.createElement('div');
    room.style.position = 'absolute';
    room.style.inset = '80px 80px 40px 40px';
    room.style.background = 'transparent';
    this.rootElement.appendChild(room);

    const floor = document.createElement('div');
    floor.style.position = 'absolute';
    floor.style.left = '60px';
    floor.style.right = '80px';
    floor.style.bottom = '60px';
    floor.style.height = '320px';
    floor.style.backgroundColor = '#3f382c';
    floor.style.backgroundImage = 'url("./assets/factory/tiles/floor.png")';
    floor.style.backgroundSize = '220px 220px';
    floor.style.backgroundRepeat = 'repeat';
    floor.style.borderRadius = '18px';
    floor.style.transform = 'perspective(800px) rotateX(55deg)';
    floor.style.transformOrigin = 'bottom center';
    floor.style.boxShadow = '0 10px 20px rgba(0,0,0,0.22)';
    room.appendChild(floor);

    const backWall = document.createElement('div');
    backWall.style.position = 'absolute';
    backWall.style.left = '70px';
    backWall.style.top = '60px';
    backWall.style.right = '120px';
    backWall.style.height = '220px';
    backWall.style.backgroundColor = '#4c4438';
    backWall.style.backgroundImage = 'url("./assets/factory/tiles/structure-wall.png")';
    backWall.style.backgroundSize = 'cover';
    backWall.style.borderRadius = '12px';
    backWall.style.boxShadow = 'inset 0 0 0 2px rgba(240,230,211,0.12)';
    room.appendChild(backWall);

    const sideWall = document.createElement('div');
    sideWall.style.position = 'absolute';
    sideWall.style.right = '80px';
    sideWall.style.top = '90px';
    sideWall.style.width = '180px';
    sideWall.style.height = '200px';
    sideWall.style.backgroundColor = '#5a4c3b';
    sideWall.style.backgroundImage = 'url("./assets/factory/decals/structure-window.png")';
    sideWall.style.backgroundSize = 'cover';
    sideWall.style.borderRadius = '12px';
    sideWall.style.boxShadow = 'inset 0 0 0 2px rgba(240,230,211,0.12)';
    room.appendChild(sideWall);

    const lamp = document.createElement('div');
    lamp.style.position = 'absolute';
    lamp.style.top = '30px';
    lamp.style.left = 'calc(50% - 40px)';
    lamp.style.width = '80px';
    lamp.style.height = '80px';
    lamp.style.borderRadius = '50%';
    lamp.style.background = 'radial-gradient(circle, #f5c882 0%, rgba(245, 200, 130, 0.45) 55%, transparent 70%)';
    room.appendChild(lamp);

    const indicator = document.createElement('div');
    indicator.style.position = 'absolute';
    indicator.style.top = '92px';
    indicator.style.left = '120px';
    indicator.style.width = '90px';
    indicator.style.height = '90px';
    indicator.style.backgroundImage = 'url("./assets/factory/decals/indicator-special-area.png")';
    indicator.style.backgroundSize = 'cover';
    indicator.style.opacity = '0.9';
    room.appendChild(indicator);

    const desks = [
      { left: '40px', top: '190px' },
      { left: '220px', top: '190px' },
      { left: '400px', top: '190px' },
      { left: '580px', top: '190px' },
      { left: '760px', top: '190px' },
    ];
    const agentRoles = {
      manager: { name: 'Manager', color: '#7ec8e3' },
      financial: { name: 'Finance', color: '#8fbc8f' },
      calendar: { name: 'Calendar', color: '#e8a85d' },
      email: { name: 'Email', color: '#f0e6d3' },
      researcher: { name: 'Research', color: '#c4a882' },
    };

    desks.forEach((desk, index) => {
      const deskEl = document.createElement('div');
      deskEl.style.position = 'absolute';
      deskEl.style.left = desk.left;
      deskEl.style.top = desk.top;
      deskEl.style.width = '120px';
      deskEl.style.height = '100px';
      deskEl.style.borderRadius = '12px';
      deskEl.style.background = '#8b6f4e';
      deskEl.style.border = '2px solid rgba(240, 230, 211, 0.25)';
      deskEl.style.boxShadow = '0 4px 14px rgba(0,0,0,0.2)';
      deskEl.style.transform = 'perspective(700px) rotateX(12deg)';
      room.appendChild(deskEl);

      const deskLabel = document.createElement('div');
      deskLabel.style.position = 'absolute';
      deskLabel.style.left = '8px';
      deskLabel.style.top = '8px';
      deskLabel.style.color = '#f0e6d3';
      deskLabel.style.fontSize = '12px';
      deskLabel.textContent = agentRoles[Object.keys(agentRoles)[index]]?.name || `Desk ${index + 1}`;
      deskEl.appendChild(deskLabel);
    });

    const conveyor = document.createElement('div');
    conveyor.style.position = 'absolute';
    conveyor.style.left = 'calc(50% - 180px)';
    conveyor.style.bottom = '80px';
    conveyor.style.width = '260px';
    conveyor.style.height = '92px';
    conveyor.style.backgroundImage = 'url("./assets/factory/props/conveyor.png")';
    conveyor.style.backgroundSize = 'cover';
    conveyor.style.opacity = '0.95';
    room.appendChild(conveyor);

    const machine = document.createElement('div');
    machine.style.position = 'absolute';
    machine.style.left = 'calc(50% + 70px)';
    machine.style.bottom = '120px';
    machine.style.width = '160px';
    machine.style.height = '140px';
    machine.style.backgroundImage = 'url("./assets/factory/props/machine.png")';
    machine.style.backgroundSize = 'contain';
    machine.style.backgroundRepeat = 'no-repeat';
    machine.style.opacity = '0.95';
    room.appendChild(machine);

    const scanner = document.createElement('div');
    scanner.style.position = 'absolute';
    scanner.style.left = '40px';
    scanner.style.bottom = '40px';
    scanner.style.width = '140px';
    scanner.style.height = '140px';
    scanner.style.backgroundImage = 'url("./assets/factory/props/scanner-low.png")';
    scanner.style.backgroundSize = 'contain';
    scanner.style.backgroundRepeat = 'no-repeat';
    scanner.style.opacity = '0.95';
    room.appendChild(scanner);

    const crate = document.createElement('div');
    crate.style.position = 'absolute';
    crate.style.left = '720px';
    crate.style.bottom = '70px';
    crate.style.width = '90px';
    crate.style.height = '90px';
    crate.style.backgroundImage = 'url("./assets/factory/props/box-small.png")';
    crate.style.backgroundSize = 'contain';
    crate.style.backgroundRepeat = 'no-repeat';
   (crate).style.opacity = '0.95';
    room.appendChild(crate);

    const robot = document.createElement('div');
    robot.style.position = 'absolute';
    robot.style.right = '110px';
    robot.style.bottom = '70px';
    robot.style.width = '120px';
    robot.style.height = '120px';
    robot.style.backgroundImage = 'url("./assets/factory/props/robot-arm-a.png")';
    robot.style.backgroundSize = 'contain';
    robot.style.backgroundRepeat = 'no-repeat';
    robot.style.opacity = '0.95';
    room.appendChild(robot);

    this.agentLayer = document.createElement('div');
    this.agentLayer.style.position = 'absolute';
    this.agentLayer.style.inset = '0';
    this.agentLayer.style.pointerEvents = 'none';
    room.appendChild(this.agentLayer);
  }

  handleSocketMessage(payload) {
    if (payload.type === 'state_update') {
      const previousAgents = this.state.agents || [];
      const nextAgents = (payload.state.agents || []).map((incomingAgent) => {
        const previousAgent = previousAgents.find((agent) => agent.id === incomingAgent.id);
        const hasAssistantThought = previousAgent && typeof previousAgent.thoughts === 'string' && !previousAgent.thoughts.includes('Local routine');
        if (!hasAssistantThought) {
          return incomingAgent;
        }
        return {
          ...incomingAgent,
          ...previousAgent,
          status: previousAgent.status || incomingAgent.status,
          thoughts: previousAgent.thoughts || incomingAgent.thoughts,
          task: previousAgent.task || incomingAgent.task,
          task_progress: previousAgent.task_progress ?? incomingAgent.task_progress,
        };
      });
      this.state = { agents: nextAgents };
      this.syncAgents();
      this.uiOverlay.setAgentsState(nextAgents);
    }
    if (payload.type === 'status_update') {
      this.uiOverlay.setStatus(payload.message || 'Assistant is working locally');
      const agentStates = payload.agent_states || {};
      const nextAgents = (this.state.agents || []).map((agent) => {
        const state = agentStates[agent.id];
        if (!state) {
          return agent;
        }
        return {
          ...agent,
          status: state.status || agent.status,
          thoughts: state.thoughts || agent.thoughts,
          task: payload.current_task || agent.task,
          task_progress: Math.round((payload.progress || 0) * 100) || agent.task_progress,
        };
      });
      this.state = { agents: nextAgents };
      this.syncAgents();
      this.uiOverlay.setAgentsState(nextAgents);
    }
    if (payload.type === 'connection' && payload.connected === false) {
      this.uiOverlay.setStatus('Reconnecting to local backend…');
    }
    if (payload.type === 'connection' && payload.connected === true) {
      this.uiOverlay.setStatus('Connected to local backend');
    }
  }

  createAgentElement(agent) {
    const container = document.createElement('div');
    container.style.position = 'absolute';
    container.style.width = '72px';
    container.style.height = '96px';
    container.style.transition = 'left 0.4s ease, top 0.4s ease, transform 0.4s ease';
    container.style.transform = 'translate(-50%, -50%)';
    container.style.pointerEvents = 'none';

    const shadow = document.createElement('div');
    shadow.style.position = 'absolute';
    shadow.style.left = '14px';
    shadow.style.bottom = '0';
    shadow.style.width = '44px';
    shadow.style.height = '12px';
    shadow.style.borderRadius = '50%';
    shadow.style.background = 'rgba(0,0,0,0.28)';
    shadow.style.filter = 'blur(2px)';
    container.appendChild(shadow);

    const figure = document.createElement('div');
    figure.style.position = 'absolute';
    figure.style.left = '0';
    figure.style.right = '0';
    figure.style.top = '8px';
    figure.style.height = '74px';
    figure.style.borderRadius = '14px 14px 10px 10px';
    figure.style.background = '#7ec8e3';
    figure.style.boxShadow = '0 6px 12px rgba(0,0,0,0.18)';
    figure.style.border = '2px solid rgba(240,230,211,0.4)';
    container.appendChild(figure);

    const head = document.createElement('div');
    head.style.position = 'absolute';
    head.style.left = '18px';
    head.style.top = '0';
    head.style.width = '24px';
    head.style.height = '24px';
    head.style.borderRadius = '50%';
    head.style.background = '#f0e6d3';
    head.style.border = '2px solid rgba(43,45,49,0.25)';
    container.appendChild(head);

    const body = document.createElement('div');
    body.style.position = 'absolute';
    body.style.left = '18px';
    body.style.top = '24px';
    body.style.width = '24px';
    body.style.height = '28px';
    body.style.borderRadius = '8px';
    body.style.background = '#6b4f3c';
    body.style.border = '2px solid rgba(240,230,211,0.3)';
    container.appendChild(body);

    const leg = document.createElement('div');
    leg.style.position = 'absolute';
    leg.style.left = '22px';
    leg.style.top = '50px';
    leg.style.width = '8px';
    leg.style.height = '22px';
    leg.style.borderRadius = '4px';
    leg.style.background = '#2b2d31';
    container.appendChild(leg);

    const label = document.createElement('div');
    label.style.position = 'absolute';
    label.style.top = '100px';
    label.style.left = '50%';
    label.style.transform = 'translateX(-50%)';
    label.style.fontSize = '10px';
    label.style.color = '#f0e6d3';
    label.style.textAlign = 'center';
    label.style.whiteSpace = 'nowrap';
    container.appendChild(label);

    return container;
  }

  updateAgentElement(element, agent) {
    const deskIndex = Math.max(0, Math.min(4, (agent.desk || 1) - 1));
    const baseX = 150 + deskIndex * 140;
    const baseY = 180;
    const x = baseX + (agent.position?.x ?? 0) * 70;
    const y = baseY - (agent.position?.y ?? 0) * 90 + (agent.status === 'working' ? -10 : 0);

    element.style.left = `${x}px`;
    element.style.top = `${y}px`;

    const roleColors = {
      manager: '#7ec8e3',
      financial: '#8fbc8f',
      calendar: '#e8a85d',
      email: '#f0e6d3',
      researcher: '#c4a882',
    };
    const colorMap = {
      idle: roleColors[agent.id] || '#7ec8e3',
      working: '#f5c882',
      sleeping: '#4d3b2e',
      alert: '#ff7f50',
    };
    const accent = colorMap[agent.status] || roleColors[agent.id] || '#7ec8e3';
    const body = element.children[1];
    const head = element.children[2];
    const leg = element.children[4];

    body.style.background = accent;
    head.style.background = agent.status === 'working' ? '#f0e6d3' : '#f0e6d3';
    leg.style.transform = agent.status === 'working' ? 'translateY(-3px)' : 'translateY(0)';
    element.style.transform = `translate(-50%, -50%) ${agent.status === 'working' ? 'scale(1.03)' : 'scale(1)'}`;

    const label = element.children[5];
    label.textContent = `${agent.name}\n${agent.status}`;
  }

  syncAgents() {
    const agents = this.state.agents || [];
    for (const agent of agents) {
      if (!this.agentElements.has(agent.id)) {
        const element = this.createAgentElement(agent);
        this.agentLayer.appendChild(element);
        this.agentElements.set(agent.id, element);
      }

      const element = this.agentElements.get(agent.id);
      this.updateAgentElement(element, agent);
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const root = document.getElementById('game');
  if (root) {
    new OfficeViewer(root);
  }
});
