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
        const data = await callDetectAPI(response.pageText);
        await updateResults(data);
            
      } catch (error) {
        console.error('Error:', error);
        results.innerHTML = `<div class="error">Error: ${error.message}</div>`;
      } finally {
        loading.classList.add('hidden');
        results.classList.remove('hidden');
      }
    });
  
    async function updateResults(data) {
        console.log('Received detection results:', data);
        
        // 更新状态和图标
        const isScam = data.scam_detected;
        statusText.textContent = isScam ? 'Potential Scam Detected!' : 'No Scam Detected';
        statusText.className = isScam ? 'warning' : 'safe';
        statusIcon.innerHTML = isScam ? '⚠️' : '✅';
        
        // 更新冰淇淋状态
        updateGelatoState(isScam);
        
        // 更新报告内容
        scamReport.innerHTML = `
            <p><strong>Analysis:</strong> ${data.report}</p>
            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
        `;
        
        // 如果检测到诈骗且有关键词，进行高亮
        if (isScam && data.matched_keywords && data.matched_keywords.length > 0) {
            console.log('Highlighting keywords:', data.matched_keywords);
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tab) {
                chrome.tabs.sendMessage(tab.id, {
                    action: 'highlightScam',
                    matches: data.matched_keywords
                });
            }
        }
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

    async function callDetectAPI(text) {
        const maxRetries = 3;
        let retryCount = 0;
        
        while (retryCount < maxRetries) {
            try {
                const response = await fetch('http://localhost:8000/detect', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                return await response.json();
            } catch (error) {
                retryCount++;
                console.error(`API call attempt ${retryCount} failed:`, error);
                if (retryCount === maxRetries) {
                    throw new Error('API connection failed after multiple attempts');
                }
                // 等待1秒后重试
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
    }
  });