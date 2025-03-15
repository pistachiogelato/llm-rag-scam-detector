"""
本地钓鱼短信生成器 - 优化增强版
优势：高可控性、低成本、生成速度快
"""

import pandas as pd
import random
from faker import Faker
from typing import List, Tuple
import re
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache

# 初始化配置
fake = Faker()
logging.basicConfig(level=logging.INFO)

class AdvancedPhishingGenerator:
    def __init__(self):
        # 多层级模板系统
        self.templates = {
            'government': [
                {
                    "template": "[{agency}] {username}: {status} {benefit} of {amount}. {action} {deadline}: {link}",
                    "slots": ["agency", "username", "status", "benefit", "amount", "action", "deadline", "link"],
                    "variants": {
                        "status": ["Unclaimed", "Pending", "Approved", "Overdue"],
                        "action": ["Claim by", "Respond before", "Validate until", "Secure your"]
                    }
                }
            ],
            'financial': [
                {
                    "template": "{alert_type}: {amount} {transaction_type} from {account}. {instruction} {link}",
                    "slots": ["alert_type", "amount", "transaction_type", "account", "instruction", "link"],
                    "variants": {
                        "alert_type": ["Urgent", "Security Alert", "Fraud Warning", "Account Notification"],
                        "transaction_type": ["transfer", "withdrawal", "payment", "charge"]
                    }
                }
            ],
            'shipping': [
                {
                    "template": "{courier}: Package #{id} requires {action}. {instruction} {link}",
                    "slots": ["courier", "id", "action", "instruction", "link"],
                    "variants": {
                        "action": ["customs clearance", "address confirmation", "payment verification"],
                        "instruction": ["Track at", "Confirm via", "Update information at"]
                    }
                }
            ]
        }

        # 增强型动态词库
        self.dynamic_dict = {
            'agency': [
                ("GOV", "Department of Treasury", "Social Security Administration"),
                ("[IRS Alert]", "[State Tax Office]", "Medicare Services")
            ],
            'benefit': [
                ("tax refund", "stimulus check", "child tax credit"),
                ("energy subsidy", "housing grant", "pandemic relief")
            ],
            'account': lambda: f"Account ****{fake.random_int(1000,9999)}"
        }

        # URL混淆模式
        self.url_patterns = [
            self._generate_obfuscated_url,
            self._generate_misspelled_url,
            self._generate_redirect_url
        ]

    @lru_cache(maxsize=1000)
    def _cached_word(self):
        """缓存常用词提升性能"""
        return fake.word()

    def _generate_username(self) -> str:
        """生成拟真用户名"""
        patterns = [
            lambda: f"{fake.first_name()} {fake.last_name()}",
            lambda: f"{self._cached_word()}{fake.random_int(1,99)}",
            lambda: f"user_{fake.random_int(100000,999999)}"
        ]
        return random.choice(patterns)()

    def _generate_obfuscated_url(self) -> str:
        """生成混淆URL"""
        base = f"{random.choice(['https://','http://',''])}{self._cached_word()}"
        tld = random.choice(['com', 'net', 'org', 'info'])
        return f"{base}.{tld}".replace('.', '[.]', 1)

    def _generate_misspelled_url(self) -> str:
        """生成拼写错误URL"""
        domain = self._cached_word()
        # 引入常见拼写错误
        errors = {
            'o': '0',
            'i': '1',
            'e': '3',
            'a': '4'
        }
        return f"{''.join([errors.get(c,c) for c in domain[:4]])}{domain[4:]}.{random.choice(['com','net'])}"

    def _generate_redirect_url(self) -> str:
        """生成重定向URL"""
        params = f"?id={fake.uuid4()}" if random.random() > 0.5 else ""
        return f"{self._cached_word()}-redirect.{random.choice(['com','net'])}{params}"

    def _semantic_enhancement(self, text: str) -> str:
        """语义增强处理"""
        # 同义词替换
        synonym_map = {
            'claim': ['collect', 'access', 'secure'],
            'verify': ['confirm', 'validate', 'authenticate']
        }
        for word, replacements in synonym_map.items():
            if word in text.lower():
                text = text.replace(word, random.choice(replacements), 1)
        
        # 句式变换
        if random.random() > 0.5:
            text = re.sub(r"(\d+)", lambda m: f"${m.group(1)}", text)
        if ':' in text:
            text = text.replace(':', random.choice([' -', '!', ' =>']), 1)
        
        return text

    def _generate_message(self) -> Tuple[str, str]:
        """单条消息生成管道"""
        try:
            category = random.choice(list(self.templates.keys()))
            template_config = random.choice(self.templates[category])
            
            params = {}
            for slot in template_config["slots"]:
                if slot in template_config.get("variants", {}):
                    params[slot] = random.choice(template_config["variants"][slot])
                elif slot in self.dynamic_dict:
                    params[slot] = random.choice(self.dynamic_dict[slot]) if isinstance(self.dynamic_dict[slot], list) \
                                  else self.dynamic_dict[slot]()
                elif slot == 'link':
                    params[slot] = random.choice(self.url_patterns)()
                elif slot == 'amount':
                    params[slot] = f"${fake.random_int(100, 999)}"
                elif slot == 'id':
                    params[slot] = f"#{fake.random_int(100000,999999)}"
                else:
                    params[slot] = self._cached_word()

            message = template_config["template"].format(**params)
            message = self._semantic_enhancement(message)
            
            # 添加自然错误
            if random.random() > 0.8:
                message = message.replace(' ', '  ', 1)
            if random.random() > 0.9:
                message = message.upper()[:len(message)//2] + message.lower()[len(message)//2:]
            
            return ('spam', message)
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return None

    def generate_batch(self, num: int) -> pd.DataFrame:
        """批量生成"""
        with ThreadPoolExecutor() as executor:
            results = list(filter(None, executor.map(lambda _: self._generate_message(), range(num))))
        
        df = pd.DataFrame(results, columns=['type', 'message'])
        return df.drop_duplicates()

if __name__ == "__main__":
    generator = AdvancedPhishingGenerator()
    
    # 生成4000条样本
    logging.info("开始生成数据集...")
    dataset = generator.generate_batch(4000)
    
    # 保存结果
    output_path = "data/balanced_sms_dataset.csv"
    dataset.to_csv(output_path, index=False)
    logging.info(f"成功生成 {len(dataset)} 条样本，保存至 {output_path}")
    
    # 打印示例
    print("\n生成示例：")
    print(dataset.sample(5).to_markdown(index=False))