class ScamDetector {
    constructor() {
      this.debounceTimeout = null;
      this.gelatoIcon = null;
      this.reportPanel = null;
      this.currentSelection = '';
      this.isAnalyzing = false;
      this.eventListeners = []; // Stores event listeners for cleanup
  
      this.init();
    }
  
    init() {
      this.createGelatoIcon();
      this.createReportPanel();
      this.setupEventListeners();
    }
  
    createGelatoIcon() {
      try {
        this.gelatoIcon = document.createElement('div');
        this.gelatoIcon.className = 'gelato-icon';
        this.gelatoIcon.innerHTML = '🍦';
        this.gelatoIcon.style.display = 'none';
        document.body.appendChild(this.gelatoIcon);
      } catch (error) {
        console.error('Error in createGelatoIcon:', error);
      }
    }
  
    createReportPanel() {
      try {
        this.reportPanel = document.createElement('div');
        this.reportPanel.className = 'report-panel';
        this.reportPanel.innerHTML = `
          <h2>Risk Analysis Report</h2>
          <div id="report-content"></div>
        `;
        document.body.appendChild(this.reportPanel);
      } catch (error) {
        console.error('Error in createReportPanel:', error);
      }
    }
  
    setupEventListeners() {
      const selectionHandler = () => {
        if (this.debounceTimeout) clearTimeout(this.debounceTimeout);
        this.debounceTimeout = setTimeout(() => this.handleSelection(), 300);
      };
      document.addEventListener('selectionchange', selectionHandler);
      this.eventListeners.push({ element: document, event: 'selectionchange', handler: selectionHandler });
  
      const iconClickHandler = () => this.analyzeText();
      this.gelatoIcon.addEventListener('click', iconClickHandler);
      this.eventListeners.push({ element: this.gelatoIcon, event: 'click', handler: iconClickHandler });
  
      const documentClickHandler = (e) => {
        if (!this.reportPanel.contains(e.target) && !this.gelatoIcon.contains(e.target)) {
          this.reportPanel.classList.remove('active');
        }
      };
      document.addEventListener('click', documentClickHandler);
      this.eventListeners.push({ element: document, event: 'click', handler: documentClickHandler });
  
      // Use pagehide instead of unload to avoid deprecation warning.
      window.addEventListener('pagehide', () => this.cleanup());
    }
  
    handleSelection() {
      try {
        const selection = window.getSelection();
        const text = selection.toString().trim();
  
        // Reset icon to default on new selection
        if (this.gelatoIcon) {
          this.gelatoIcon.className = 'gelato-icon';
          this.gelatoIcon.innerHTML = '🍦';
        }
  
        if (text.length < 50 || text.length > 2000) {
          this.hideGelatoIcon();
          return;
        }
  
        if (this.isCodeOrStyle(selection.anchorNode)) {
          this.hideGelatoIcon();
          return;
        }
  
        this.currentSelection = text;
        this.updateGelatoPosition(selection);
      } catch (error) {
        console.error('Error in handleSelection:', error);
      }
    }
  
    isCodeOrStyle(node) {
      const invalidTags = ['SCRIPT', 'STYLE', 'CODE', 'PRE'];
      let current = node;
      while (current && current.nodeType === Node.ELEMENT_NODE) {
        if (invalidTags.includes(current.tagName)) return true;
        current = current.parentElement;
      }
      return false;
    }
  
    updateGelatoPosition(selection) {
      try {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
  
        if (this.gelatoIcon) {
          let left = rect.right + 20;
          let top = rect.bottom + window.scrollY + 20;
  
          if (left + 80 > viewportWidth) {
            left = rect.left - 100;
          }
          if (top + 80 > window.scrollY + viewportHeight) {
            top = rect.top + window.scrollY - 100;
          }
          left = Math.max(20, Math.min(left, viewportWidth - 100));
          top = Math.max(20, top);
  
          this.gelatoIcon.style.transition = 'all 0.3s ease';
          this.gelatoIcon.style.left = `${left}px`;
          this.gelatoIcon.style.top = `${top}px`;
          this.gelatoIcon.style.display = 'block';
        }
      } catch (error) {
        console.error('Error in updateGelatoPosition:', error);
      }
    }
  
    hideGelatoIcon() {
      if (this.gelatoIcon) {
        this.gelatoIcon.style.display = 'none';
      }
    }
  
    async analyzeText() {
      if (this.isAnalyzing) return;
      this.isAnalyzing = true;
  
      try {
        // Fade out current icon
        if (this.gelatoIcon) {
          this.gelatoIcon.style.transition = 'opacity 0.3s ease';
          this.gelatoIcon.style.opacity = '0';
          await this.delay(300);
          this.gelatoIcon.className = 'gelato-icon analyzing';
          this.gelatoIcon.innerHTML = '⏳';
          this.gelatoIcon.style.opacity = '1';
        }
  
        const response = await fetch('http://localhost:8000/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: this.currentSelection })
        });
  
        if (!response.ok) throw new Error('API request failed');
        const data = await response.json();
        await this.delay(500);
        await this.handleAnalysisResult(data);
  
      } catch (error) {
        console.error('Error in analyzeText:', error);
        this.showError();
      }
    }
  
    delay(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }
  
    async handleAnalysisResult(data) {
      try {
        const riskScore = parseFloat(data.risk_score);
        const riskLevel = this.calculateRiskLevel(riskScore);
        await this.showReport(data, riskLevel);
        await this.delay(300);
        await this.updateGelatoState(riskLevel, riskScore);
      } catch (error) {
        console.error('Error in handleAnalysisResult:', error);
      } finally {
        this.isAnalyzing = false;
      }
    }
  
    calculateRiskLevel(score) {
      if (score >= 0.7) return 'critical';
      if (score >= 0.5) return 'high';
      if (score >= 0.3) return 'medium';
      return 'low';
    }
  
    async updateGelatoState(riskLevel, score) {
      try {
        if (this.gelatoIcon) {
          const newState = this.getGelatoState(riskLevel, score);
          this.gelatoIcon.style.opacity = '0';
          await this.delay(300);
          this.gelatoIcon.className = `gelato-icon ${newState.className}`;
          this.gelatoIcon.innerHTML = newState.icon;
          this.gelatoIcon.style.opacity = '1';
        }
      } catch (error) {
        console.error('Error in updateGelatoState:', error);
      }
    }
  
    getGelatoState(riskLevel, score) {
      switch (riskLevel) {
        case 'critical':
          return { icon: '🍦🔥 DANGER!', className: 'danger' };
        case 'high':
          return { icon: '🍦⚠️ Warning!', className: 'warning' };
        case 'medium':
          return { icon: '🍦❗ Stay Alert!', className: 'caution' };
        default:
          return { icon: '🍦✨ Safe', className: '' };
      }
    }
  
    async showReport(data, riskLevel) {
      try {
        // Display only risk level, risk score, and recommended actions
        const content = `
          <div class="report-header ${riskLevel}-risk">
            <h3 class="risk-title">
              ${this.getRiskLevelEmoji(riskLevel)} 
              ${this.getRiskLevelText(riskLevel)}
            </h3>
            <div class="confidence-score">
              <span>Risk Score:</span>
              <strong>${(data.risk_score * 100).toFixed(1)}%</strong>
            </div>
          </div>
  
          <div class="report-body">
            ${this.getRecommendationsHTML(data)}
            
            <div class="report-footer">
              ${this.getFooterMessage(riskLevel)} ${this.getFooterEmoji(riskLevel)}
            </div>
          </div>
        `;
  
        const reportContent = document.getElementById('report-content');
        if (reportContent) {
          reportContent.innerHTML = content;
          this.reportPanel.classList.add('active');
        }
      } catch (error) {
        console.error('Error in showReport:', error);
      }
    }
  
    getRecommendationsHTML(data) {
      if (data.llm_analysis && data.llm_analysis.recommendations && data.llm_analysis.recommendations.length > 0) {
        let html = '<div class="report-section"><h4>Recommended Actions</h4><ul class="recommendations-list">';
        data.llm_analysis.recommendations.forEach(action => {
          html += `<li class="recommendation-item">${this.escapeHtml(action)}</li>`;
        });
        html += '</ul></div>';
        return html;
      }
      return '<div class="report-section"><h4>No recommendations provided</h4></div>';
    }
  
    getRiskLevelEmoji(riskLevel) {
      switch (riskLevel) {
        case 'critical': return '🚨';
        case 'high': return '⚠️';
        case 'medium': return '❗';
        default: return '✨';
      }
    }
  
    getRiskLevelText(riskLevel) {
      switch (riskLevel) {
        case 'critical': return 'Critical Risk - Melting Alert!';
        case 'high': return 'High Risk - Handle with Care!';
        case 'medium': return 'Medium Risk - Stay Alert!';
        default: return 'Low Risk - Enjoy Safely!';
      }
    }
  
    getFooterMessage(riskLevel) {
      switch (riskLevel) {
        case 'critical': return 'Emergency! Your gelato is melting fast!';
        case 'high': return 'Careful! This gelato needs attention!';
        case 'medium': return 'Keep an eye on your gelato!';
        default: return 'Enjoy your delicious gelato!';
      }
    }
  
    getFooterEmoji(riskLevel) {
      switch (riskLevel) {
        case 'critical': return '🍦💦🔥';
        case 'high': return '🍦⚠️';
        case 'medium': return '🍦❗';
        default: return '🍦✨🎉';
      }
    }
  
    escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }
  
    showError() {
      try {
        if (this.gelatoIcon) {
          this.gelatoIcon.innerHTML = '🍨💧';
          this.gelatoIcon.className = 'gelato-icon';
        }
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = 'Analysis failed. Please try again.';
        document.body.appendChild(tooltip);
        const rect = this.gelatoIcon.getBoundingClientRect();
        tooltip.style.left = `${rect.left}px`;
        tooltip.style.top = `${rect.bottom + 10}px`;
        setTimeout(() => {
          if (tooltip.parentNode) tooltip.parentNode.removeChild(tooltip);
        }, 3000);
      } catch (error) {
        console.error('Error in showError:', error);
      }
    }
  
    cleanup() {
      try {
        this.eventListeners.forEach(({ element, event, handler }) => {
          element.removeEventListener(event, handler);
        });
        this.eventListeners = [];
        if (this.gelatoIcon && this.gelatoIcon.parentNode) {
          this.gelatoIcon.parentNode.removeChild(this.gelatoIcon);
        }
        if (this.reportPanel && this.reportPanel.parentNode) {
          this.reportPanel.parentNode.removeChild(this.reportPanel);
        }
      } catch (error) {
        console.error('Error in cleanup:', error);
      }
    }
  }
  
  window.addEventListener('load', () => {
    try {
      new ScamDetector();
    } catch (error) {
      console.error('Error initializing ScamDetector:', error);
    }
  });
  
  // (Optional) Helper function to extract visible main content.
  function getVisibleMainContent() {
    const mainContentTags = [
      'body', 'article', 'main', 'div[role="main"]',
      '.main-content', '#main-content', 'p', 'section', '.content', '#content'
    ];
    const viewportHeight = window.innerHeight;
    const texts = [];
    let hasContent = false;
    for (const tag of mainContentTags) {
      const elements = document.querySelectorAll(tag);
      elements.forEach(element => {
        const rect = element.getBoundingClientRect();
        if (rect.top >= -viewportHeight && rect.bottom <= viewportHeight * 2) {
          const text = element.textContent.trim();
          if (text.length > 10) {
            texts.push(text);
            hasContent = true;
          }
        }
      });
      if (hasContent) break;
    }
    if (!hasContent) {
      return document.body.textContent.trim();
    }
    return texts.join('\n');
  }
  