"""
从tiny_shakespeare数据集自动提取角色信息
这个脚本分析tiny_shakespeare数据集，提取角色信息，并为每个角色创建上下文摘要
"""
from datasets import load_dataset
import re
import json
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import spacy
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import torch

# 下载NLTK资源（如果尚未下载）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 加载spaCy模型用于命名实体识别
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("下载spaCy模型...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# 莎士比亚主要作品列表
SHAKESPEARE_PLAYS = [
    "hamlet", "macbeth", "othello", "king lear", "romeo and juliet", 
    "merchant of venice", "midsummer night's dream", "twelfth night",
    "tempest", "julius caesar", "antony and cleopatra", "much ado about nothing"
]

# 已知的主要莎士比亚角色（用于验证）
KNOWN_CHARACTERS = [
    "hamlet", "ophelia", "claudius", "gertrude", "polonius", "horatio", "laertes",
    "king lear", "cordelia", "goneril", "regan", "edmund", "edgar",
    "romeo", "juliet", "mercutio", "tybalt", "friar lawrence", "nurse",
    "macbeth", "lady macbeth", "banquo", "macduff", "three witches",
    "othello", "desdemona", "iago", "cassio", "emilia",
    "prospero", "miranda", "ariel", "caliban",
    "shylock", "portia", "antonio", "bassanio",
    "falstaff", "puck", "oberon", "titania"
]

def load_shakespeare_data():
    """加载tiny_shakespeare数据集"""
    print("加载tiny_shakespeare数据集...")
    dataset = load_dataset("tiny_shakespeare")
    return dataset

def extract_character_mentions(text):
    """
    从文本中提取角色提及
    使用spaCy进行命名实体识别并过滤出人物
    """
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return persons

def find_character_passages(text, character_name, context_window=200):
    """
    查找文本中提及特定角色的段落
    返回包含角色名的上下文片段
    """
    # 不区分大小写的搜索
    pattern = re.compile(r'\b' + re.escape(character_name) + r'\b', re.IGNORECASE)
    
    passages = []
    for match in pattern.finditer(text):
        start = max(0, match.start() - context_window)
        end = min(len(text), match.end() + context_window)
        passage = text[start:end]
        # 确保片段在句子边界上
        if start > 0:
            first_period = passage.find('.')
            if first_period > 0:
                passage = passage[first_period + 1:]
        if end < len(text):
            last_period = passage.rfind('.')
            if last_period > 0:
                passage = passage[:last_period + 1]
        
        passages.append(passage.strip())
    
    return passages

def summarize_character(character_name, passages, summarizer=None):
    """
    使用预训练模型总结角色的信息
    基于提取的包含该角色的文本片段
    """
    if not passages:
        return f"Character {character_name} appears in Shakespeare's works but no detailed information available."
    
    # 如果没有提供摘要器，使用简单连接
    if summarizer is None:
        # 选择最具代表性的片段（这里简单地选择前3个）
        representative_passages = passages[:3]
        return " ".join(representative_passages)
    
    # 使用提供的摘要器
    combined_text = " ".join(passages[:5])  # 限制长度以避免超出模型限制
    summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    return summary

def generate_character_description(character_name, passages, gpt2_model=None, gpt2_tokenizer=None):
    """
    使用GPT-2模型根据文本片段生成莎士比亚风格的角色描述
    """
    if not gpt2_model or not gpt2_tokenizer or not passages:
        return None
    
    try:
        # 选择较短的代表性片段
        combined_text = " ".join(passages[:2])
        if len(combined_text) > 500:
            combined_text = combined_text[:500]
        
        # 创建莎士比亚风格的提示
        prompt = f"In the style of William Shakespeare, describe the character {character_name} based on these passages: {combined_text}\n\nShakespeare's description of {character_name}:"
        
        # 生成描述
        input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
        
        # 将输入移动到正确的设备
        device = next(gpt2_model.parameters()).device
        input_ids = input_ids.to(device)
        
        # 生成描述
        with torch.no_grad():  # 禁用梯度计算
            output = gpt2_model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=gpt2_tokenizer.eos_token_id,
                attention_mask=input_ids.ne(gpt2_tokenizer.eos_token_id)
            )
        
        # 将输出移回CPU进行解码
        output = output.cpu()
        description = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 提取生成的描述部分
        description = description.split(f"Shakespeare's description of {character_name}:")[-1].strip()
        
        return description
        
    except Exception as e:
        print(f"生成描述时出错: {e}")
        return None

def extract_and_organize_character_info():
    """
    主函数：提取和组织莎士比亚角色信息
    """
    # 加载数据集
    dataset = load_shakespeare_data()
    
    # 获取文本内容
    text = dataset["train"]["text"][0]
    
    # 初始化结果字典
    character_info = {}
    
    # 拆分文本为更小的段落处理
    print("将文本拆分为段落...")
    paragraphs = re.split(r'\n\n+', text)
    
    # 使用spaCy提取人物名称
    print("提取人物名称...")
    extracted_characters = set()
    for para in paragraphs:
        characters = extract_character_mentions(para)
        extracted_characters.update([c.lower() for c in characters])
    
    # 过滤出已知的莎士比亚角色
    confirmed_characters = [char for char in KNOWN_CHARACTERS if char in extracted_characters]
    
    print(f"找到 {len(confirmed_characters)} 个莎士比亚角色")
    
    # 初始化摘要器
    try:
        print("加载摘要模型...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"加载摘要模型失败: {e}")
        summarizer = None
    
    # 尝试加载GPT-2模型用于生成莎士比亚风格的描述
    try:
        print("加载GPT-2模型...")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # 检查 MPS 可用性并设置设备
        if torch.backends.mps.is_available():
            print("使用 MPS 设备")
            device = torch.device("mps")
        else:
            print("使用 CPU 设备")
            device = torch.device("cpu")
        
        # 将模型移动到设备
        gpt2_model = gpt2_model.to(device)
        gpt2_model.eval()  # 设置为评估模式
        
    except Exception as e:
        print(f"加载GPT-2模型失败: {e}")
        gpt2_tokenizer = None
        gpt2_model = None
    
    # 为每个角色收集信息
    print("为每个角色收集信息...")
    for character in confirmed_characters:
        print(f"处理角色: {character}")
        
        try:
            # 查找提及该角色的段落
            passages = find_character_passages(text, character)
            
            if passages:
                # 识别角色所属的作品
                play = "unknown"
                for p in SHAKESPEARE_PLAYS:
                    if p.lower() in text.lower():
                        play = p
                        break
                
                # 生成描述
                description = generate_character_description(
                    character, passages, gpt2_model, gpt2_tokenizer
                )
                
                # 存储信息
                character_info[character] = {
                    "play": play,
                    "description": description,
                    "passages": passages[:3]  # 只保存前3个段落
                }
                
        except Exception as e:
            print(f"处理角色 {character} 时出错: {e}")
            continue
    
    # 保存结果
    with open("shakespeare_characters.json", "w", encoding="utf-8") as f:
        json.dump(character_info, f, ensure_ascii=False, indent=2)
    
    print("角色信息已保存到 shakespeare_characters.json")
    return character_info

if __name__ == "__main__":
    extract_and_organize_character_info()
