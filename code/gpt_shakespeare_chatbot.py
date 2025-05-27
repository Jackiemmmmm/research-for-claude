"""
增强版莎士比亚聊天机器人 - 使用从tiny_shakespeare提取的角色信息
"""

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import json
import os
from datetime import datetime

# 模型路径 - 使用增强版模型
MODEL_PATH = "./gpt2-shakespeare-final"
# 角色信息JSON文件
CHARACTER_INFO_PATH = "shakespeare_characters.json"

# 初始化角色信息
try:
    with open(CHARACTER_INFO_PATH, "r") as f:
        CHARACTER_INFO = json.load(f)
except FileNotFoundError:
    st.warning(
        "找不到角色信息文件，将使用有限的角色列表。请先运行 extract_shakespeare_characters.py 生成角色信息。"
    )
    CHARACTER_INFO = {}

# 莎士比亚作品的主要角色列表
SHAKESPEARE_CHARACTERS = [
    "hamlet",
    "ophelia",
    "claudius",
    "gertrude",
    "polonius",
    "horatio",
    "laertes",
    "king lear",
    "cordelia",
    "goneril",
    "regan",
    "edmund",
    "edgar",
    "romeo",
    "juliet",
    "mercutio",
    "tybalt",
    "friar lawrence",
    "nurse",
    "macbeth",
    "lady macbeth",
    "banquo",
    "macduff",
    "three witches",
    "othello",
    "desdemona",
    "iago",
    "cassio",
    "emilia",
    "prospero",
    "miranda",
    "ariel",
    "caliban",
    "shylock",
    "portia",
    "antonio",
    "bassanio",
    "falstaff",
    "puck",
    "oberon",
    "titania",
]


# 加载模型和分词器
@st.cache_resource
def load_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        # 回退到原始GPT-2模型
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        return tokenizer, model


# 检测用户输入是否询问了莎士比亚角色
def detect_character_query(user_input):
    # 将输入转为小写以进行不区分大小写的匹配
    lower_input = user_input.lower()

    # 检测常见的询问模式
    query_patterns = [
        r"who\s+(?:is|are|was)\s+([a-z\s]+)",
        r"tell\s+(?:me|us)\s+about\s+([a-z\s]+)",
        r"what\s+(?:do|can|could)\s+(?:you|thou)\s+(?:know|tell)\s+about\s+([a-z\s]+)",
        r"(?:describe|explain)\s+([a-z\s]+)",
        r"(?:who|what)\s+(?:is|are|was)\s+(?:the\s+character\s+)?([a-z\s]+)",
    ]

    for pattern in query_patterns:
        matches = re.findall(pattern, lower_input)
        for match in matches:
            match = match.strip()
            # 检查匹配项是否是莎士比亚角色
            if any(
                character == match or character in match
                for character in SHAKESPEARE_CHARACTERS
            ):
                # 尝试找到完全匹配
                for character in SHAKESPEARE_CHARACTERS:
                    if character == match or character in match:
                        return True, character

    # 检查输入中是否直接包含任何角色名
    for character in SHAKESPEARE_CHARACTERS:
        if character in lower_input:
            return True, character

    return False, None


# 根据提取的角色信息创建提示
def create_character_prompt(character):
    """根据提取的角色信息创建莎士比亚风格的提示"""
    if character in CHARACTER_INFO:
        info = CHARACTER_INFO[character]
        name = info.get("name", character.title())
        play = info.get("play", "mine plays")

        # 优先使用莎士比亚风格的描述
        if info.get("shakespeare_description"):
            description = info["shakespeare_description"]
        else:
            description = info.get("summary", f"a character from {play}")

        # 如果有引用段落，选择一个添加进去
        passages = info.get("passages", [])
        passage_quote = f'As I once penned: "{passages[0]}"' if passages else ""

        prompt = f"""
        The query doth concern {name}, a character from mine own play known as {play}. 
        {description}
        
        {passage_quote}
        
        Pray, speak of this character with the depth and eloquence thy pen hath given them in mine own work.
        Use thine Elizabethan language and poetic flourish to describe {name}'s nature, role, and significance.
        """
    else:
        # 对于没有提取信息的角色，使用通用提示
        prompt = f"""
        The query doth concern {character}, one of the characters from mine own plays. 
        Pray, describe this personage as I would in mine own plays, with depth and poetic flourish. 
        Speaketh of their nature, their role, and their significance in the drama.
        """

    return prompt.strip()


# 生成莎士比亚风格的回复
def generate_shakespeare_response(user_input, tokenizer, model):
    # 检测是否询问了莎士比亚角色
    is_character_query, character = detect_character_query(user_input)

    # 设计提示词，包含风格指导和角色信息
    system_prompt = "Thou art William Shakespeare, the greatest playwright and poet of the English language. Speak in thine own voice, with thy characteristic eloquence, rich vocabulary, and poetic rhythm. Use 'thee', 'thou', 'thy', 'mine', 'methinks', and other Elizabethan English terms."

    character_prompt = ""
    if is_character_query and character:
        character_prompt = create_character_prompt(character)

    # 组合提示词
    full_prompt = (
        f"{system_prompt}\n{character_prompt}\nUser: {user_input}\nShakespeare:"
    )

    # 编码提示词
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

    # 确保输入不会太长
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, -1024:]

    # 生成回复
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=min(
                    input_ids.shape[1] + 200, 1024
                ),  # 增加长度但不超过模型限制
                do_sample=True,
                temperature=0.7,  # 降低温度以增加连贯性
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.2,  # 添加重复惩罚
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # 防止重复n-gram
            )

        # 解码生成的文本
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 提取回复部分
        response = output_text[len(full_prompt) :].strip()

        # 如果回复包含多行，只取第一段
        if "\n\n" in response:
            response = response.split("\n\n")[0]

        # 确保回复不为空
        if not response:
            response = (
                "Forsooth, mine mind wanders. What wouldst thou know of me or my works?"
            )

        # 验证角色回复质量
        if is_character_query and character:
            if len(response) < 50 or not any(
                term in response.lower()
                for term in ["thee", "thou", "thy", "hath", "doth"]
            ):
                # 回退到预定义的高质量回复
                response = get_fallback_character_response(character)

        return response

    except Exception as e:
        st.error(f"生成回复时发生错误: {e}")
        return "Alas, my quill hath faltered. Pray, ask again thy question."


# 获取角色的预定义备用回复
def get_fallback_character_response(character):
    """为主要角色提供预定义的高质量备用回复"""
    FALLBACK_RESPONSES = {
        "hamlet": "Alas, Hamlet! The melancholy Prince of Denmark, mine own creation that hath captured the hearts of audiences through centuries. This noble youth, son to the late King Hamlet and nephew to the usurping Claudius, doth struggle with a burden most terrible - the ghost of his father commanding revenge. 'To be or not to be,' he ponders, trapped betwixt duty and doubt, action and contemplation. A soul of infinite jest and most excellent fancy, yet consumed by grief and rage. His tragedy unfolds as he feigns madness, rejects his beloved Ophelia, and ultimately achieves his vengeance, though at terrible cost to all. In Hamlet, I sought to paint the very depths of human contemplation and the price of vengeance.",
        "ophelia": "Ah, Ophelia! A delicate flower in the harsh winter of Denmark's court. Daughter to Polonius and sister to Laertes, her gentle heart was bound to Hamlet, my tragic prince. Yet fate, cruel mistress, did condemn her love. First by her father's stern command, then by Hamlet's feigned madness and cruel rejection. Poor Ophelia, driven to true madness by grief when her father fell by Hamlet's sword! Her piteous end, drowned amidst the flowers she so loved, hath moved many a heart to tears. 'There is a willow grows aslant a brook,' where she, garland in hand, did meet her watery fate. Such fragile beauty, crushed betwixt the wheels of vengeance and power.",
        "othello": "Noble Othello, the Moor of Venice! A general of matchless valor whose sword hath won many a battle for Venice. Yet his heart, so steadfast in war, proved vulnerable in love. Married to fair Desdemona against her father's will, his soul was poisoned by the serpent Iago's whispers of jealousy. O, how this mighty warrior, unfamiliar with the deceits of Venice, was undone by doubt! 'Not poppy, nor mandragora, nor all the drowsy syrups of the world' could medicine him to that sweet sleep he knew before jealousy infected his mind. In rage most terrible, he smothered his innocent wife, only to learn too late of her fidelity. Pierced by truth and shame, he fell upon his sword, kissing his beloved in death as in life. A noble heart, yet undone by the green-eyed monster, jealousy.",
        "macbeth": "Brave Macbeth! Once the most valiant captain in Scotland's army, whose courage in battle won him the Thane of Cawdor's title. Yet ambition, that sin by which angels fell, did corrupt his noble nature when three weird sisters prophesied his ascent to Scotland's throne. Goaded by his lady's sharp ambition, he murdered good King Duncan, a guest beneath his roof. 'Will all great Neptune's ocean wash this blood clean from my hand?' he cried, but once embarked upon the bloody path, he waded deeper still. Banquo, Macduff's family, all fell to secure his ill-got crown. Yet conscience made a coward of this brave man, haunted by Banquo's ghost and his lady's madness. 'Tomorrow, and tomorrow, and tomorrow,' he lamented, as life became a tale told by an idiot, full of sound and fury, signifying nothing. His tale doth show how vaulting ambition o'erleaps itself and falls on th'other side.",
        "romeo": "Young Romeo, scion of House Montague and star-crossed lover in Verona's ancient feud! How swiftly did his heart turn from pining for cold Rosaline to burning for sweet Juliet of the rival house. 'What light through yonder window breaks?' he sighed beneath her balcony, as love transformed this melancholy youth to one who dared defy family, fortune, and fate for his beloved. Alas, when banished for Tybalt's death, cruel circumstance and hasty passion led him to poison himself beside Juliet's seeming corpse. In death, these lovers purchased peace for Verona with the price of their blood, ending the ancient grudge with their parents' late-found mercy. A youth of fire and tenderness, whose impassioned heart both doomed and immortalized him.",
        "juliet": "Sweet Juliet, my youngest heroine, scarce fourteen summers old when fate and fair Verona cast her in love's tragedy! Daughter to rich Capulet, promised to County Paris, yet her heart awoke to love when first she gazed on Romeo. 'What's in a name?' she pondered, rejecting the ancient hatred between their houses, choosing love o'er duty, passion o'er obedience. With courage beyond her tender years, she drank the friar's potion, gambling death for love. Awaking to find her Romeo cold in death beside her, she joined him with his dagger, proving 'the fearful passage of their death-marked love.' Never was there tale of more woe than this of Juliet and her Romeo.",
    }

    # 尝试从提取的角色信息中获取备用回复
    if character in CHARACTER_INFO and CHARACTER_INFO[character].get(
        "shakespeare_description"
    ):
        description = CHARACTER_INFO[character]["shakespeare_description"]
        name = CHARACTER_INFO[character].get("name", character.title())

        # 确保描述采用莎士比亚风格
        if any(
            term in description.lower()
            for term in ["thee", "thou", "thy", "hath", "doth"]
        ):
            return f"Ah, {name}! {description}"

    # 使用预定义回复
    if character in FALLBACK_RESPONSES:
        return FALLBACK_RESPONSES[character]

    # 通用备用回复
    return f"Pray, {character} is a character of mine creation, whose tale unfolds upon the stage with passions deep and motives true to human nature. Wouldst thou hear more of this personage in particular? Thou might inquire of their role or deeds within the play where they dwell."


# 保存用户反馈
def save_feedback(user_input, response, feedback, improvement=None):
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "model_response": response,
        "feedback": feedback,
        "improvement": improvement,
    }

    # 确保文件夹存在
    os.makedirs("feedback", exist_ok=True)

    # 将反馈添加到文件
    with open("feedback/user_feedback.jsonl", "a") as f:
        f.write(json.dumps(feedback_data) + "\n")


# 处理示例问题点击
def handle_example_click(question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 添加用户消息到历史
    st.session_state.chat_history.append(("你", question))

    # 加载模型
    tokenizer, model = load_model()

    # 生成莎士比亚风格的回复
    response = generate_shakespeare_response(question, tokenizer, model)

    # 添加响应到历史
    st.session_state.chat_history.append(("莎士比亚", response))


# 主应用
def main():
    st.title("🎭 莎士比亚风格聊天机器人")
    st.markdown(
        """
    > *"To be, or not to be, that is the question..."*
    
    与莎士比亚本人对话！询问他关于他的作品、角色或任何话题。他会以优雅的伊丽莎白时代英语回答你。
    """
    )

    # 加载模型
    tokenizer, model = load_model()

    # 初始化聊天历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 初始化反馈状态
    if "show_improvement_feedback" not in st.session_state:
        st.session_state.show_improvement_feedback = False

    # 显示当前聊天历史
    for speaker, message in st.session_state.chat_history:
        if speaker == "你":
            st.markdown(f"**👤 你:** {message}")
        else:
            st.markdown(f"**🎭 莎士比亚:** {message}")

    # 用户输入
    with st.form(key="user_input_form"):
        user_input = st.text_input("你想对莎士比亚说点什么？")
        submit_button = st.form_submit_button("发送")

        if submit_button and user_input:
            # 添加用户消息到历史
            st.session_state.chat_history.append(("你", user_input))

            # 生成莎士比亚风格的回复
            with st.spinner("莎士比亚正在思考..."):
                response = generate_shakespeare_response(user_input, tokenizer, model)

            # 添加响应到历史
            st.session_state.chat_history.append(("莎士比亚", response))

            # 重新加载页面以显示新消息
            st.rerun()

    # 用户反馈机制
    if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
        st.markdown("---")
        st.markdown("### 你觉得这个回答怎么样？")

        last_user_input = ""
        last_response = ""

        # 获取最后一次交互
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            if st.session_state.chat_history[i][0] == "莎士比亚":
                last_response = st.session_state.chat_history[i][1]
                # 查找对应的用户输入
                if i > 0 and st.session_state.chat_history[i - 1][0] == "你":
                    last_user_input = st.session_state.chat_history[i - 1][1]
                break

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 很好的回答"):
                save_feedback(last_user_input, last_response, "positive")
                st.success("感谢您的反馈！")

        with col2:
            if st.button("👎 需要改进"):
                st.session_state.show_improvement_feedback = True
                st.rerun()

        # 显示改进反馈
