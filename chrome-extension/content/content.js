// 确保content script正确加载
console.log('Content script loaded');

// 添加消息监听器
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Message received in content script:', request);
  
  if (request.action === 'scanPage') {
    try {
      const pageText = getPageText();
      console.log('Page text extracted, length:', pageText.length);
      sendResponse({ pageText });
    } catch (error) {
      console.error('Error in content script:', error);
      sendResponse({ error: error.message });
    }
  }
  
  if (request.action === 'highlightScam') {
    highlightScamText(request.matches);
  }
  
  return true;  // 保持消息通道开放
});

function getPageText() {
  // 排除脚本和样式标签
  const blacklist = ['script', 'style', 'noscript', 'iframe'];
  const texts = [];
  
  function extractText(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent.trim();
      if (text) texts.push(text);
    } else if (
      node.nodeType === Node.ELEMENT_NODE && 
      !blacklist.includes(node.tagName.toLowerCase())
    ) {
      Array.from(node.childNodes).forEach(extractText);
    }
  }
  
  extractText(document.body);
  return texts.join(' ');
}
/*
function highlightScamText(matches) {
  // 创建一个包含所有可疑文本的正则表达式
  const pattern = new RegExp(matches.map(text => 
    text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'gi');
  
  // 遍历文本节点并高亮匹配内容
  function walkText(node) {
    if (node.nodeType === 3) {
      const text = node.textContent;
      const matches = text.match(pattern);
      
      if (matches) {
        const span = document.createElement('span');
        span.innerHTML = text.replace(pattern, match => 
          `<mark class="scam-highlight">${match}</mark>`);
        node.parentNode.replaceChild(span, node);
      }
    } else {
      Array.from(node.childNodes).forEach(walkText);
    }
  }
  
  walkText(document.body);
}*/
function highlightScamText(keywords) {
  // 移除现有的高亮
  document.querySelectorAll('.scam-highlight').forEach(el => {
      const parent = el.parentNode;
      parent.replaceChild(document.createTextNode(el.textContent), el);
  });
  
  // 为每个关键词创建正则表达式
  const patterns = keywords.map(keyword => 
      new RegExp(`(${keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  );
  
  // 遍历文本节点并高亮匹配内容
  function walkText(node) {
      if (node.nodeType === 3) {
          let text = node.textContent;
          let matched = false;
          
          patterns.forEach(pattern => {
              if (pattern.test(text)) {
                  matched = true;
                  text = text.replace(pattern, match => 
                      `<mark class="scam-highlight" data-keyword="${match}">${match}</mark>`
                  );
              }
          });
          
          if (matched) {
              const span = document.createElement('span');
              span.innerHTML = text;
              node.parentNode.replaceChild(span, node);
          }
      } else if (node.nodeType === 1 && 
                !['SCRIPT', 'STYLE', 'MARK'].includes(node.tagName)) {
          Array.from(node.childNodes).forEach(walkText);
      }
  }
  
  walkText(document.body);
}