// 监听安装事件
chrome.runtime.onInstalled.addListener(() => {
  console.log('Scam Detector Extension installed');
});

// 可以添加其他后台功能，如：
// - 定期检查更新
// - 管理缓存
// - 处理通知等
