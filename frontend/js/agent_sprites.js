class AgentSprite extends Phaser.GameObjects.Container {
  constructor(scene, x, y, agentData) {
    super(scene, x, y);
    this.agentData = agentData;
    this.body = scene.add.rectangle(0, 0, 34, 44, 0x7ec8e3, 0.95);
    this.body.setOrigin(0.5, 0.5);
    this.body.setStrokeStyle(2, 0xf5c882);
    this.body.setRadius(8);

    this.face = scene.add.rectangle(0, -4, 24, 16, 0x2b2d31, 0.95);
    this.face.setOrigin(0.5, 0.5);
    this.face.setStrokeStyle(1, 0xf5c882);

    this.eyeLeft = scene.add.rectangle(-6, -4, 4, 4, 0xf0e6d3, 0.95);
    this.eyeRight = scene.add.rectangle(6, -4, 4, 4, 0xf0e6d3, 0.95);

    this.add(this.body);
    this.add(this.face);
    this.add(this.eyeLeft);
    this.add(this.eyeRight);

    this.setSize(34, 44);
    scene.add.existing(this);

    this.tween = null;
    this.setAgentState(agentData.status || 'idle');
  }

  setAgentState(status) {
    const colorMap = {
      idle: 0x7ec8e3,
      walking: 0x8b6f4e,
      working: 0x6b4f3c,
      sleeping: 0x4d3b2e,
      alert: 0xf5c882,
    };
    this.body.setFillStyle(colorMap[status] || 0x7ec8e3);
  }

  updateFromState(agent) {
    this.agentData = agent;
    this.setAgentState(agent.status || 'idle');
    if (this.tween) {
      this.tween.stop();
    }

    const targetX = agent.position?.x ?? 0;
    const targetY = agent.position?.y ?? 0;
    this.tween = this.scene.tweens.add({
      targets: this,
      x: targetX * 80,
      y: targetY * 80,
      duration: 650,
      ease: 'Sine.easeInOut',
      onComplete: () => {
        this.tween = null;
      },
    });
  }
}
