import re
import pandas as pd
from ollama import chat
from process_csvs import retrieve_rag

class NutritionAssistant:
    def __init__(self, model_name):
        self.model = model_name
        self.profile = {
            "Nome": None,
            "Idade": None,                     # e.g., "22 anos"
            "Género": None,
            "Altura": None,                    # e.g., "175 cm"
            "Peso": None,                      # e.g., "68 kg"
            "Nível de Atividade Física": None,
            "Alergias": None,
            "Intolerâncias": None,
            "Condições Médicas": None,
            "Preferências Dietéticas": None,
            "Objetivos de Saúde": None
        }
        self.history = []
        self.fields_order = list(self.profile.keys())
        self.current_field_index = 0

        # Default units for fields
        self.default_units = {
            "Idade": "anos",
            "Altura": "cm",
            "Peso": "kg"
        }

    def extract_nutrient(self, text, nutrient):
        match = re.search(rf"{nutrient}: ([0-9.]+)", text)
        return match.group(1) if match else "N/A"


    def next_field(self):
        while self.current_field_index < len(self.fields_order):
            field = self.fields_order[self.current_field_index]
            if not self.profile[field]:
                return field
            self.current_field_index += 1
        return None

    def update_profile_field(self, field, value):
        v = value.strip()
        # Add default unit if missing
        if field in self.default_units:
            unit = self.default_units[field]
            if not re.search(rf"\b{unit}\b", v, re.IGNORECASE):
                v = f"{v} {unit}"
        self.profile[field] = v

    def detect_update_command(self, user_input):
        """
        Matches phrases like:
         - "atualiza o meu peso para 66 kg"
         - "podes atualizar minha altura para 180 cm?"
        """
        pattern = re.compile(
            r"\b(?:atualiz(?:ar|e|a))\b.*?\b("
            + "|".join([re.escape(f) for f in self.profile.keys()]) +
            r")\b.*?\b(?:para|com)\b\s*(?P<value>[\d.,]+\s*\w+)",
            flags=re.IGNORECASE
        )
        match = pattern.search(user_input)
        if match:
            field = match.group(1).strip().capitalize()
            value = match.group("value").strip()
            for key in self.profile:
                if key.lower() == field.lower():
                    return key, value
        return None, None

    def profile_block(self):
        lines = ["Perfil do Utilizador:"]
        for k, v in self.profile.items():
            if v:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines) + "\n\n" if any(self.profile.values()) else ""

    def sanitize_response(self, text):
        disclaimer = (
            "\n\n**Isto não substitui uma consulta com um nutricionista ou profissional de saúde. "
            "Para orientações clínicas específicas, consulte um especialista.**"
        )
        if disclaimer not in text:
            text += disclaimer
        forbidden = ["alcoólico", "cocktail", "prescrição farmacológica"]
        if any(word in text.lower() for word in forbidden):
            return "Desculpe, o meu foco restringe-se apenas em nutrição e hábitos saudáveis." + disclaimer
        return text

    def ask(self, user_input: str) -> str:
        # 1) Handle profile updates
        if self.next_field() is None:
            field, value = self.detect_update_command(user_input)
            if field:
                self.update_profile_field(field, value)
                reply = f"{field} atualizado para {self.profile[field]}."
                self.history.extend([("user", user_input), ("assistant", reply)])
                return reply

        # 2) Collect profile fields
        field = self.next_field()
        if field:
            self.update_profile_field(field, user_input)
            self.history.append(("user", user_input))
            self.current_field_index += 1
            next_f = self.next_field()
            if next_f:
                reply = f"Por favor, indica o teu(a) **{next_f}**:"
            else:
                reply = "Obrigado! Agora que tenho o teu perfil, em que posso ajudar-te em nutrição?"
            self.history.append(("assistant", reply))
            return reply

        # 3) Profile complete → perform RAG retrieval
        docs = retrieve_rag(user_input, k=3)
        rag_ctx = "\n\n".join(
            f"Alimento: {row.description}\n"
            f"Proteína: {self.extract_nutrient(row.text, 'Protein')}g\n"
            f"Calorias: {self.extract_nutrient(row.text, 'Energy')}kcal"
            for _, row in docs.iterrows()
        )

        print(f"RAGXTX:{rag_ctx}")

        # 4) Build messages with profile and RAG context
        messages = [
            {"role": "system", "content": self.profile_block()},
            {"role": "system", "content": "Use APENAS os dados nutricionais fornecidos abaixo:\n\n" + rag_ctx},
            {"role": "user", "content": user_input}
        ]

        # 5) Query LLM
        response = chat(model=self.model, messages=messages)
        reply = self.sanitize_response(response["message"]["content"])
        self.history.extend([("user", user_input), ("assistant", reply)])
        return reply

    def save_history(self, filename="conversation_history.txt"):
        with open(filename, "w", encoding="utf-8") as f:
            for role, message in self.history:
                f.write(f"{role.title()}: {message}\n\n")

if __name__ == "__main__":
    assistant = NutritionAssistant("nutri-assistant-v2:latest")
    print("Olá! Sou o teu Assistente de Nutrição. Escreve 'sair' para terminar.")
    next_field = assistant.next_field()
    print(f"Assistente: Por favor, indica o teu(a) **{next_field}**:")

    while True:
        user_input = input("Tu: ").strip()
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Assistente: Bom apetite e até breve!")
            assistant.save_history()
            break
        reply = assistant.ask(user_input)
        print(f"Assistente: {reply}\n")

    print("Histórico guardado em 'conversation_history.txt'.")
