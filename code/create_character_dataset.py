"""
创建包含莎士比亚角色信息的数据集，用于增强模型的知识
"""
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

# 创建莎士比亚角色数据集
characters_data = [
    {"character": "Hamlet", "description": "Prince of Denmark, protagonist in the tragedy Hamlet. Melancholic, philosophical, and contemplating revenge for his father's murder."},
    {"character": "Ophelia", "description": "Hamlet's love interest who goes mad and drowns after her father Polonius is killed by Hamlet."},
    {"character": "Claudius", "description": "Hamlet's uncle who murdered Hamlet's father and married his mother to become king of Denmark."},
    {"character": "Gertrude", "description": "Hamlet's mother who married Claudius shortly after her husband's death."},
    {"character": "Polonius", "description": "Father of Ophelia and Laertes, counsellor to the king who is accidentally killed by Hamlet."},
    {"character": "Horatio", "description": "Hamlet's loyal friend who lives to tell Hamlet's story."},
    {"character": "Laertes", "description": "Son of Polonius who seeks revenge against Hamlet for his father's death."},
    
    {"character": "King Lear", "description": "Aging king who divides his kingdom between two of his three daughters based on their flattery, leading to tragedy."},
    {"character": "Cordelia", "description": "King Lear's youngest daughter who refuses to flatter him but truly loves him."},
    {"character": "Goneril", "description": "King Lear's eldest daughter who betrays him after gaining power."},
    {"character": "Regan", "description": "King Lear's middle daughter who joins with Goneril against their father."},
    {"character": "Edmund", "description": "Illegitimate son of Gloucester who plots against his legitimate brother Edgar."},
    {"character": "Edgar", "description": "Legitimate son of Gloucester who disguises himself as Poor Tom."},
    
    {"character": "Romeo", "description": "Young man from the Montague family who falls in love with Juliet from the rival Capulet family."},
    {"character": "Juliet", "description": "Young woman from the Capulet family who falls in love with Romeo from the rival Montague family."},
    {"character": "Mercutio", "description": "Romeo's friend who is killed by Tybalt, leading to Romeo's revenge and banishment."},
    {"character": "Tybalt", "description": "Juliet's cousin who kills Mercutio and is killed by Romeo in revenge."},
    {"character": "Friar Lawrence", "description": "Priest who secretly marries Romeo and Juliet and plans their escape."},
    {"character": "Nurse", "description": "Juliet's devoted nurse and confidante who helps arrange her marriage to Romeo."},
    
    {"character": "Macbeth", "description": "Scottish general who becomes king through murder after hearing prophecies from three witches."},
    {"character": "Lady Macbeth", "description": "Macbeth's wife who encourages him to murder King Duncan but later goes mad with guilt."},
    {"character": "Banquo", "description": "Macbeth's friend who is murdered on Macbeth's orders after witches prophesy his descendants will be kings."},
    {"character": "Macduff", "description": "Scottish nobleman who kills Macbeth at the end of the play, fulfilling the prophecy."},
    {"character": "Three Witches", "description": "Supernatural beings who prophesy that Macbeth will become king, setting the tragedy in motion."},
    
    {"character": "Othello", "description": "Moorish general in the Venetian army who is manipulated into believing his wife is unfaithful."},
    {"character": "Desdemona", "description": "Othello's wife who is wrongly accused of infidelity and murdered by him."},
    {"character": "Iago", "description": "Othello's ensign who manipulates him into believing Desdemona is unfaithful out of jealousy and hatred."},
    {"character": "Cassio", "description": "Othello's lieutenant whom Iago uses in his plot against Othello."},
    {"character": "Emilia", "description": "Iago's wife and Desdemona's attendant who exposes her husband's plot too late."},
    
    {"character": "Prospero", "description": "Rightful Duke of Milan and powerful magician in The Tempest who orchestrates events to reclaim his dukedom."},
    {"character": "Miranda", "description": "Prospero's daughter who falls in love with Ferdinand after being raised in isolation on an island."},
    {"character": "Ariel", "description": "Spirit servant to Prospero who helps him execute his plans in exchange for freedom."},
    {"character": "Caliban", "description": "Deformed island native enslaved by Prospero who attempts to overthrow him."},
    
    {"character": "Shylock", "description": "Jewish moneylender in The Merchant of Venice who demands a pound of flesh as payment for a loan."},
    {"character": "Portia", "description": "Wealthy heiress in The Merchant of Venice who disguises herself as a lawyer to save Antonio."},
    {"character": "Antonio", "description": "Merchant of Venice who borrows money from Shylock for his friend Bassanio."},
    {"character": "Bassanio", "description": "Antonio's friend who borrows money to woo Portia."},
    
    {"character": "Falstaff", "description": "Fat, boastful, and cowardly knight who appears in multiple plays including Henry IV and The Merry Wives of Windsor."},
    {"character": "Puck", "description": "Mischievous fairy in A Midsummer Night's Dream who carries out Oberon's orders and creates chaos."},
    {"character": "Oberon", "description": "King of the fairies in A Midsummer Night's Dream who quarrels with his queen Titania."},
    {"character": "Titania", "description": "Queen of the fairies in A Midsummer Night's Dream who falls in love with Bottom (transformed into a donkey) due to a spell."},
]

# 生成莎士比亚风格的角色描述文本
def create_shakespeare_character_texts():
    shakespeare_style_texts = []
    
    # 对每个角色创建莎士比亚风格的描述
    for char in characters_data:
        character = char["character"]
        description = char["description"]
        
        # 创建对话风格的训练文本
        query_styles = [
            f"Who is {character}?",
            f"Tell me about {character}",
            f"What dost thou know of {character}?",
            f"Pray, tell me of {character}",
            f"Speak to me of {character} from thy plays",
            f"I would hear of {character}",
        ]
        
        for query in query_styles:
            # 为每个查询创建莎士比亚风格的回复
            response = f"""
            Thou asketh of {character}? Verily, 'tis {description}
            
            In mine own words, this character doth stand prominent in my works.
            {character} doth embody the very essence of what it means to be human, 
            with all the passions, follies, and virtues that attend our mortal coil.
            
            As I have penn'd in mine own hand, such characters as these doth mirror life itself,
            holding up nature's looking glass to show virtue her own feature, scorn her own image.
            """
            
            # 使用不同格式来包装对话
            text = f"User: {query}\nShakespeare: {response.strip()}\n\n"
            shakespeare_style_texts.append({"text": text})
    
    return shakespeare_style_texts

def enhance_dataset():
    # 加载原始数据集
    original_dataset = load_dataset("tiny_shakespeare")
    
    # 创建角色数据集
    character_texts = create_shakespeare_character_texts()
    character_dataset = Dataset.from_list(character_texts)
    
    # 打印统计信息
    print(f"原始训练集大小: {len(original_dataset['train'])}")
    print(f"角色信息数据集大小: {len(character_dataset)}")
    
    # 合并数据集 - 将角色数据集添加到训练集
    # 注意：这里我们需要确保数据格式一致
    # 将原始数据集转换为与角色数据集相同的格式
    original_formatted = original_dataset["train"].map(
        lambda example: {"text": example["text"]},
        remove_columns=original_dataset["train"].column_names
    )
    
    # 合并数据集
    enhanced_dataset = concatenate_datasets([original_formatted, character_dataset])
    print(f"增强后的数据集大小: {len(enhanced_dataset)}")
    
    # 保存增强的数据集
    enhanced_dataset.save_to_disk("enhanced_shakespeare_dataset")
    print("增强的数据集已保存到 'enhanced_shakespeare_dataset'")
    
    return enhanced_dataset

if __name__ == "__main__":
    enhance_dataset()
