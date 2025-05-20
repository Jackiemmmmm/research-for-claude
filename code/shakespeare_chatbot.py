# 莎士比亚聊天机器人 - 轻量级实现
# 使用基于检索的方法和预定义模板

import streamlit as st
import random
import re
from datetime import datetime
import json
import os


class ShakespeareChatbot:
    def __init__(self):
        self.shakespeare_quotes = {
            "hamlet": [
                "To be or not to be, that is the question.",
                "There is nothing either good or bad, but thinking makes it so.",
                "Something is rotten in the state of Denmark.",
                "Though this be madness, yet there is method in't.",
                "Brevity is the soul of wit.",
            ],
            "romeo_juliet": [
                "But soft, what light through yonder window breaks?",
                "A rose by any other name would smell as sweet.",
                "These violent delights have violent ends.",
                "For never was a story of more woe than this of Juliet and her Romeo.",
                "Love is a smoke made with the fume of sighs.",
            ],
            "macbeth": [
                "Is this a dagger which I see before me?",
                "Out, damned spot! Out, I say!",
                "Double, double toil and trouble; Fire burn and caldron bubble.",
                "Tomorrow, and tomorrow, and tomorrow creeps in this petty pace.",
                "Fair is foul, and foul is fair.",
            ],
            "othello": [
                "O, beware, my lord, of jealousy! It is the green-eyed monster.",
                "She loved me for the dangers I had passed.",
                "But I will wear my heart upon my sleeve for daws to peck at.",
                "Reputation, reputation, reputation! O, I have lost my reputation!",
            ],
        }

        self.character_info = {
            "hamlet": {
                "description": "Prince of Denmark, contemplative and melancholic",
                "key_traits": [
                    "indecisive",
                    "philosophical",
                    "tormented",
                    "intelligent",
                ],
                "relationships": [
                    "Gertrude (mother)",
                    "Claudius (uncle/stepfather)",
                    "Ophelia (love interest)",
                ],
            },
            "romeo": {
                "description": "Young Montague, passionate lover",
                "key_traits": ["romantic", "impulsive", "passionate", "young"],
                "relationships": [
                    "Juliet (beloved)",
                    "Mercutio (friend)",
                    "Benvolio (cousin)",
                ],
            },
            "juliet": {
                "description": "Young Capulet, Romeo's beloved",
                "key_traits": ["intelligent", "brave", "loyal", "determined"],
                "relationships": [
                    "Romeo (beloved)",
                    "Nurse (confidante)",
                    "Lord Capulet (father)",
                ],
            },
            "macbeth": {
                "description": "Scottish general turned king through murder",
                "key_traits": ["ambitious", "brave", "guilty", "paranoid"],
                "relationships": [
                    "Lady Macbeth (wife)",
                    "Duncan (victim)",
                    "Banquo (friend/victim)",
                ],
            },
            "othello": {
                "description": "Moorish general in Venice",
                "key_traits": ["noble", "jealous", "trusting", "tragic"],
                "relationships": [
                    "Desdemona (wife)",
                    "Iago (manipulator)",
                    "Cassio (lieutenant)",
                ],
            },
        }

        self.play_summaries = {
            "hamlet": "Prince Hamlet seeks revenge against his uncle Claudius who murdered his father. The play explores themes of revenge, madness, mortality, and moral corruption.",
            "romeo_juliet": "Two young star-crossed lovers from feuding families in Verona fall in love and secretly marry, but their love ends in tragedy.",
            "macbeth": "A Scottish general, driven by ambition and his wife's encouragement, murders King Duncan and usurps the throne, leading to his downfall.",
            "othello": "A Moorish general is manipulated by his ensign Iago into believing his wife Desdemona is unfaithful, leading to tragedy.",
        }

        self.scene_summaries = {
            "hamlet_act3_scene1": "Hamlet contemplates suicide in his famous 'To be or not to be' soliloquy, then confronts Ophelia about love and marriage.",
            "romeo_juliet_act2_scene2": "The famous balcony scene where Romeo and Juliet profess their love and decide to marry.",
            "macbeth_act5_scene1": "Lady Macbeth sleepwalks and reveals her guilt over the murders through her unconscious confessions.",
            "othello_act5_scene2": "Othello kills Desdemona in their bedroom, then learns of Iago's deception and kills himself.",
        }

        self.conversation_history = []

    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        self.conversation_history.append(
            {"user": user_input, "timestamp": datetime.now()}
        )

        # 问候处理
        if any(greeting in user_input for greeting in ["hello", "hi", "hey", "good"]):
            return self._greeting_response()

        # 引用请求
        if any(word in user_input for word in ["quote", "saying", "line"]):
            return self._get_quote(user_input)

        # 角色信息请求
        if any(word in user_input for word in ["character", "who is", "tell me about"]):
            return self._get_character_info(user_input)

        # 剧情摘要请求
        if any(word in user_input for word in ["summary", "about", "plot", "story"]):
            return self._get_play_summary(user_input)

        # 场景摘要请求
        if any(word in user_input for word in ["scene", "act"]):
            return self._get_scene_summary(user_input)

        # 主题讨论
        if any(word in user_input for word in ["theme", "meaning", "represents"]):
            return self._discuss_themes(user_input)

        # 帮助信息
        if "help" in user_input:
            return self._get_help()

        # 默认响应
        return self._default_response()

    def _greeting_response(self):
        greetings = [
            "Hail and well met! I am thy humble guide to the works of the immortal Bard, William Shakespeare.",
            "Good morrow! How may I assist thee in exploring the timeless tales of Shakespeare?",
            "Greetings, fair scholar! What wisdom from Shakespeare's quill dost thou seek?",
        ]
        return random.choice(greetings)

    def _get_quote(self, user_input):
        # 检查是否指定了特定剧作
        for play in self.shakespeare_quotes:
            if play.replace("_", " ") in user_input or play in user_input:
                quote = random.choice(self.shakespeare_quotes[play])
                return f"From {play.replace('_', ' ').title()}: \"{quote}\""

        # 返回随机引用
        all_quotes = []
        for quotes in self.shakespeare_quotes.values():
            all_quotes.extend(quotes)
        quote = random.choice(all_quotes)
        return f'Here\'s a famous line from Shakespeare: "{quote}"'

    def _get_character_info(self, user_input):
        for character in self.character_info:
            if character in user_input:
                info = self.character_info[character]
                response = f"{character.title()}: {info['description']}\n"
                response += f"Key traits: {', '.join(info['key_traits'])}\n"
                response += (
                    f"Important relationships: {', '.join(info['relationships'])}"
                )
                return response

        return "I can tell thee about characters like Hamlet, Romeo, Juliet, Macbeth, or Othello. Which one interests thee?"

    def _get_play_summary(self, user_input):
        for play in self.play_summaries:
            if play.replace("_", " ") in user_input or play in user_input:
                return f"{play.replace('_', ' ').title()}: {self.play_summaries[play]}"

        response = "I can provide summaries of these great works:\n"
        for play, summary in self.play_summaries.items():
            response += f"• {play.replace('_', ' ').title()}\n"
        return response

    def _get_scene_summary(self, user_input):
        # 检查特定场景
        for scene_key in self.scene_summaries:
            scene_parts = scene_key.split("_")
            if all(part in user_input for part in scene_parts[1:]):  # 检查act和scene
                play_name = scene_parts[0].replace("_", " ").title()
                act_scene = " ".join(scene_parts[1:]).title()
                return f"{play_name}, {act_scene}: {self.scene_summaries[scene_key]}"

        return "I can summarize famous scenes. Try asking about specific acts and scenes, like 'Hamlet Act 3 Scene 1' or 'Romeo and Juliet Act 2 Scene 2'."

    def _discuss_themes(self, user_input):
        themes = {
            "love": "Love in Shakespeare ranges from the pure passion of Romeo and Juliet to the manipulated trust between Othello and Desdemona.",
            "revenge": "Revenge drives much of Shakespeare's tragedy, most notably in Hamlet's quest to avenge his father.",
            "ambition": "Unchecked ambition leads to downfall, as seen in Macbeth's rise and fall.",
            "jealousy": "Jealousy destroys lives, particularly evident in Othello's tragic end.",
            "power": "The corruption of power and its consequences appear throughout Shakespeare's works.",
            "death": "Death is omnipresent in Shakespeare's tragedies, often as both ending and beginning.",
            "fate": "The tension between fate and free will drives many of Shakespeare's plots.",
        }

        for theme, explanation in themes.items():
            if theme in user_input:
                return f"On the theme of {theme}: {explanation}"

        return "Shakespeare explores many themes: love, revenge, ambition, jealousy, power, death, and fate. Which would you like to discuss?"

    def _get_help(self):
        return """I can assist thee with:
        
📚 Play summaries (e.g., "Tell me about Hamlet")
👥 Character information (e.g., "Who is Romeo?")
💬 Famous quotes (e.g., "Give me a quote from Macbeth")
🎭 Scene summaries (e.g., "Hamlet Act 3 Scene 1")
🤔 Theme discussions (e.g., "What about love in Shakespeare?")

Simply ask your question in natural language!"""

    def _default_response(self):
        responses = [
            "I'm not certain I understand thy query. Could thou rephrase it?",
            "Pray tell, what specifically about Shakespeare's works interests thee?",
            "Mayhap thou could ask about a particular play, character, or quote?",
            "I am here to discuss the Bard's works. What would thou like to know?",
        ]
        return random.choice(responses)

    def get_conversation_stats(self):
        return {
            "total_exchanges": len(self.conversation_history),
            "session_start": (
                self.conversation_history[0]["timestamp"]
                if self.conversation_history
                else None
            ),
        }


# Streamlit 应用界面
def main():
    st.set_page_config(page_title="Shakespeare Chatbot", page_icon="🎭", layout="wide")

    st.title("🎭 The Bard's Companion")
    st.subheader("An Intelligent Shakespeare Chatbot")

    # 初始化聊天机器人
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ShakespeareChatbot()
        st.session_state.messages = []

    # 侧边栏信息
    with st.sidebar:
        st.header("About This Chatbot")
        st.write(
            """
        This lightweight Shakespeare chatbot can:
        - Provide famous quotes
        - Explain characters
        - Summarize plays and scenes
        - Discuss themes
        """
        )

        st.header("Example Questions")
        st.write(
            """
        - "Give me a quote from Hamlet"
        - "Who is Romeo?"
        - "Tell me about Macbeth"
        - "What's the balcony scene about?"
        - "Discuss love in Shakespeare"
        """
        )

        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.chatbot = ShakespeareChatbot()
            st.rerun()

    # 聊天界面
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("Ask about Shakespeare's works..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取机器人响应
        response = st.session_state.chatbot.get_response(prompt)

        # 添加机器人响应
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # 底部统计信息
    if st.session_state.chatbot.conversation_history:
        stats = st.session_state.chatbot.get_conversation_stats()
        st.caption(f"Conversation started: {stats['total_exchanges']} exchanges")


if __name__ == "__main__":
    main()
