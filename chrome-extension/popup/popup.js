document.addEventListener('DOMContentLoaded', function() {
    const scanButton = document.getElementById('scanPage');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    const scamReport = document.getElementById('scamReport');
    const confidence = document.getElementById('confidence');
    const statusText = document.getElementById('statusText');
    const statusIcon = document.getElementById('statusIcon');
    const gelato = document.getElementById('gelato');
    const messageContainer = document.getElementById('messageContainer');

    async function highlightScamText(text, matches) {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) return;
      
      // 发送消息给 content script 来高亮显示可疑文本
      chrome.tabs.sendMessage(tab.id, {
          action: 'highlightScam',
          matches: matches
      });
    }
    
    function updateGelatoState(isScam) {
        if (isScam) {
            gelato.classList.add('melting');
            messageContainer.className = 'message-container message-warning';
            messageContainer.textContent = 'Be cautious, your gelato is melting! 🌡️';
        } else {
            gelato.classList.remove('melting');
            messageContainer.className = 'message-container message-safe';
            messageContainer.textContent = 'Enjoy your gelato! 😊';
        }
    }


    scanButton.addEventListener('click', async () => {
      try {
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        
        // 获取当前标签页
        const [tab] = await chrome.tabs.query({ 
          active: true, 
          currentWindow: true 
        });
        
        if (!tab) {
          throw new Error('No active tab found');
        }
        
        // 注入content script
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ['content/content.js']
        });
        
        // 发送消息到content script
        const response = await new Promise((resolve, reject) => {
          chrome.tabs.sendMessage(tab.id, { action: 'scanPage' }, response => {
            if (chrome.runtime.lastError) {
              reject(new Error(chrome.runtime.lastError.message));
            } else {
              resolve(response);
            }
          });
        });
        
        if (response.error) {
          throw new Error(response.error);
        }
        
        // 调用API
        const apiResponse = await fetch('http://localhost:8000/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: response.pageText })
        });
        
        if (!apiResponse.ok) {
          throw new Error(`HTTP error! status: ${apiResponse.status}`);
        }
        
        const data = await apiResponse.json();
        updateResults(data);
        // 更新 Gelato 状态
        updateGelatoState(data.scam_detected);
            
        // 如果检测到诈骗，高亮显示可疑文本
        if (data.scam_detected && data.matched_keywords) {
            await highlightScamText(response.pageText, data.matched_keywords);
        }
        
      } catch (error) {
        console.error('Error:', error);
        results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
      } finally {
        loading.classList.add('hidden');
        results.classList.remove('hidden');
      }
    });
  
    function updateResults(data) {
      if (data.scam_detected) {
        statusText.textContent = 'Potential Scam Detected!';
        statusText.className = 'warning';
        statusIcon.innerHTML = '⚠️';
      } else {
        statusText.textContent = 'No Scam Detected';
        statusText.className = 'safe';
        statusIcon.innerHTML = '✅';
      }
  
      scamReport.innerHTML = `
        <p><strong>Analysis:</strong> ${data.report}</p>
        ${data.retrieved_cases && data.retrieved_cases.length > 0 ? `
          <p><strong>Similar cases found:</strong></p>
          <ul>
            ${data.retrieved_cases.map(caseItem => `<li>${caseItem}</li>`).join('')}
          </ul>
        ` : ''}
      `;
  
      confidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    }
    /*
    // 保存检测历史
    function saveHistory(url, result) {
      chrome.storage.local.get(['scanHistory'], function(data) {
        const history = data.scanHistory || [];
        history.unshift({
          url,
          result,
          timestamp: new Date().toISOString()
        });
        // 只保留最近50条记录
        chrome.storage.local.set({
          scanHistory: history.slice(0, 50)
        });
      });
      
    }
    */
  });