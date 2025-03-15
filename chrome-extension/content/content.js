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
      this.gelatoIcon = document.createElement('div');
      this.gelatoIcon.className = 'gelato-icon';
      this.gelatoIcon.innerHTML = '🍦';
      this.gelatoIcon.style.display = 'none';
      document.body.appendChild(this.gelatoIcon);
    }
  
    createReportPanel() {
      this.reportPanel = document.createElement('div');
      this.reportPanel.className = 'report-panel';
      this.reportPanel.innerHTML = `
        <h2>Risk Analysis Report</h2>
        <div id="report-content"></div>
      `;
      document.body.appendChild(this.reportPanel);
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
  
      // Use pagehide event instead of unload to avoid deprecation warning.
      window.addEventListener('pagehide', () => this.cleanup());
    }
  
    handleSelection() {
      try {
        const selection = window.getSelection();
        const text = selection.toString().trim();
  
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
  
        if (this.gelatoIcon) {
          this.gelatoIcon.style.left = `${rect.right + 20}px`;
          this.gelatoIcon.style.top = `${rect.bottom + window.scrollY + 20}px`;
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
      if (this.gelatoIcon) {
        this.gelatoIcon.className = 'gelato-icon analyzing';
        this.gelatoIcon.innerHTML = '⏳';
      }
  
      try {
        const response = await fetch('http://localhost:8000/detect', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ text: this.currentSelection })
        });
  
        if (!response.ok) throw new Error('API request failed');
  
        const data = await response.json();
        this.handleAnalysisResult(data);
      } catch (error) {
        console.error('Error in analyzeText:', error);
        this.showError();
      } finally {
        // Mark analysis as complete only after report is shown.
        // Do not update the icon here to ensure spinner remains until updateGelatoState runs.
        this.isAnalyzing = false;
      }
    }
  
    handleAnalysisResult(data) {
      try {
        const riskLevel = data.risk_score > 0.7 ? 'high' :
                          data.risk_score > 0.3 ? 'low' : 'safe';
  
        // Update icon based on risk and show report.
        this.updateGelatoState(riskLevel);
        this.highlightText(riskLevel);
        this.showReport(data);
      } catch (error) {
        console.error('Error in handleAnalysisResult:', error);
      }
    }
  
    updateGelatoState(riskLevel) {
      try {
        if (this.gelatoIcon) {
          this.gelatoIcon.className = `gelato-icon ${riskLevel === 'high' ? 'danger' : ''}`;
          // Update icon from spinner (⏳) to a risk-specific icon.
          this.gelatoIcon.innerHTML = riskLevel === 'high' ? '🍦🔥' :
                                       riskLevel === 'low' ? '🍦⚠️' : '🍦';
        }
      } catch (error) {
        console.error('Error in updateGelatoState:', error);
      }
    }
  
    highlightText(riskLevel) {
      if (riskLevel === 'safe') return;
  
      try {
        const range = window.getSelection().getRangeAt(0);
        const span = document.createElement('span');
        span.className = `risk-highlight ${riskLevel}`;
        range.surroundContents(span);
      } catch (error) {
        console.error('Error in highlightText:', error);
      }
    }
  
    showReport(data) {
      try {
        const content = `
          <div class="report-section">
            <h3>Risk Level: ${this.getRiskLevelText(data.risk_score)}</h3>
            <p>Confidence: ${data.confidence}</p>
          </div>
          ${data.pattern_analysis && data.pattern_analysis.matched_patterns && data.pattern_analysis.matched_patterns.length > 0 ? `
            <div class="report-section">
              <h3>Detected Patterns</h3>
              <ul>
                ${data.pattern_analysis.matched_patterns.map(p => `<li>${this.escapeHtml(p)}</li>`).join('')}
              </ul>
            </div>
          ` : ''}
          ${data.similar_texts && data.similar_texts.length > 0 ? `
            <div class="report-section">
              <h3>Similar Cases</h3>
              <ul>
                ${data.similar_texts.map(case_ => `
                  <li>
                    <strong>Similarity: ${(case_.similarity * 100).toFixed(1)}%</strong>
                    <p>${this.escapeHtml(case_.text.substring(0, 100))}...</p>
                  </li>
                `).join('')}
              </ul>
            </div>
          ` : ''}
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
  
    getRiskLevelText(score) {
      if (score >= 0.7) return 'Critical Risk 🚨';
      if (score >= 0.3) return 'Moderate Risk ⚠️';
      return 'Low Risk ✅';
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
  
  // (Optional) A helper function to extract visible main content can be defined here.
  function getVisibleMainContent() {
    // Extend text extraction scope.
    const mainContentTags = [
      'body', // Use body as fallback
      'article',
      'main',
      'div[role="main"]',
      '.main-content',
      '#main-content',
      'p',
      'section',
      '.content',
      '#content'
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
  