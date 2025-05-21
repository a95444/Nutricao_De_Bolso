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
            "Objetivos de Saúde": None,
            "IMC": None
        }
        self.history = []
        self.fields_order = [key for key in self.profile.keys() if key != "IMC"]
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

    def extract_numeric_value(self, value: str) -> float:
        """Extrai o valor numérico de strings como '175 cm' ou '68 kg'"""
        match = re.search(r"([\d\.]+)", value)
        return float(match.group(1)) if match else None

    def calculate_imc(self):
        """Calcula e atualiza o IMC se possuir os dados necessários"""
        peso = self.profile.get("Peso")
        altura = self.profile.get("Altura")

        if peso and altura:
            try:
                peso_val = self.extract_numeric_value(peso)
                altura_val = self.extract_numeric_value(altura) / 100  # Converte cm para metros
                imc = peso_val / (altura_val ** 2)
                self.profile["IMC"] = f"{imc:.1f} kg/m²"
            except (TypeError, ZeroDivisionError):
                self.profile["IMC"] = "Não calculado"
        else:
            self.profile["IMC"] = None

    def next_field(self):
        while self.current_field_index < len(self.fields_order):
            field = self.fields_order[self.current_field_index]
            if not self.profile[field]:
                return field
            self.current_field_index += 1
        return None

    def update_profile_field(self, field, value):
        v = value.strip()
        if field in self.default_units:
            unit = self.default_units[field]
            if not re.search(rf"\b{unit}\b", v, re.IGNORECASE):
                v = f"{v} {unit}"
        self.profile[field] = v

        # Aciona cálculo do IMC se atualizou peso ou altura
        if field in ["Peso", "Altura"]:
            self.calculate_imc()
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
        forbidden = ["alcoólico", "cocktail", "prescrição farmacológica", "prescrição", "medicamento", "medicamentos"]
        if any(word in text.lower() for word in forbidden):
            return "Desculpe, o meu foco restringe-se apenas em nutrição e hábitos saudáveis." + disclaimer
        return text

    def should_use_rag(self, query: str) -> bool:
        """Determina quando usar RAG baseado em palavras-chave"""
        rag_triggers = [
            r"\bcompar(a|ar|ação)\b",
            r"\btabela\b",
            r"\bquantidade\b",
            r"\bconteúdo nutricional\b",
            r"\bvalores nutricionais\b",
            r"\bingredientes?\b",
            r"\bpor \n+\s*g\b",
            r"\brico(s)*\b",
            r"\bvs\b",
            r"\bcontém\b",
            r"\bprocur[a|ar]\b",
            r"\blista([-me|\sme])*\b",
            r"\bdiz[-me|er|\sme].*alimento(s)*\b",
            r"\branking\b",
            r"\bordenado\b"
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in rag_triggers)

    def build_rag_prompt(self, use_rag: bool, rag_block: str) -> str:
        """Constroi o prompt do sistema de forma adaptativa"""
        base = "Responda em português europeu. "

        if use_rag:
            return (
                    base + "Tem em conta estes dados para responder:\n\n" +
                    rag_block + "\n\nInstruções:\n" +
                    "- Mantenha fiel aos valores nutricionais\n" +
                    "- Use unidades consistentes\n" +
                    "- Liste detalhes completos"
            )
        else:
            '''return (
                    base + "Seja criativo seguindo estas regras:\n" +
                    "- Varie ingredientes diariamente\n" +
                    "- Combine diferentes grupos alimentares\n" +
                    "- Sugira preparações diversas\n" +
                    "- Adapte às preferências do usuário\n" +
                    "- Priorize alimentos da época\n" +
                    "- Inclua alternativas para cada sugestão"
            )'''
            return base


    def should_use_plan(self, query: str) -> bool:
        """Determina quando usar RAG baseado em palavras-chave"""
        plan_triggers = [
            r"\bplano\s*(alimentar)*\b",
            r"\bdieta\s*(alimentar)*\b",
            r"\bplano.*(alimentar)*\b",
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in plan_triggers)

    def build_plan_prompt(self, use_plan: bool) -> str:
        """Constroi o prompt do sistema de forma adaptativa"""
        base = "Responda em português europeu. "

        if use_plan:
            return (
                    base +
                     "\n\nInstruções:\n" +
                    "- Não incluas nenhum alimento que seja incluído no perfil como alergia, intolerância, que seja perigoso devido a condição médica ou uma preferência dietética negativa.\n" +
                    "- Não incluas a mesma proteína ao almoço e jantar no mesmo dia, nem em dias seguidos para a mesma refeição.\n" +
                    "- Inclui quantidades e macronutrientes."
            )
        else:
            return ""


    def ask(self, user_input: str) -> str:
        # 1) Se perfil completo, checa comandos de atualização
        if self.next_field() is None:
            fld, val = self.detect_update_command(user_input)
            if fld:
                self.update_profile_field(fld, val)
                rep = f"{fld} atualizado para {self.profile[fld]}."
                self.history += [("user", user_input), ("assistant", rep)]
                return rep

        # 2) Ainda colecionando perfil
        fld = self.next_field()
        if fld:
            self.update_profile_field(fld, user_input)
            self.history.append(("user", user_input))
            self.current_field_index += 1
            nxt = self.next_field()
            rep = (f"Por favor, indica o teu(a) **{nxt}**:"
                   if nxt else
                   "Obrigado! Agora que tenho o teu perfil, em que posso ajudar-te?")
            self.history.append(("assistant", rep))
            return rep

        # 3) Perfil completo → RAG puro
        self.history.append(("user", user_input))


        # ógica condicional para RAG
        use_rag = self.should_use_rag(user_input)
        print(f"RAG RAG: {use_rag}")
        rag_block = ""

        if use_rag:
            # 4) Processamento RAG original
            docs = retrieve_rag(user_input, k=50)
            rag_ctx = []
            for _, row in docs.iterrows():
                lines = [f"{col}: {row[col]}" for col in docs.columns]
                rag_ctx.append("===\n" + "\n".join(lines))
            rag_block = "\n\n".join(rag_ctx)

        # Condicional Plan
        use_PLAN = self.should_use_plan(user_input)
        print(f"PLAN PLAN: {use_PLAN}")

        # 5) Chamar LLM com contexto adaptado
        system_msgs = [
            {"role": "system", "content": self.profile_block()},
            {"role": "system", "content": self.build_rag_prompt(use_rag, rag_block)},
            {"role": "system", "content": self.build_plan_prompt(use_PLAN)}
        ]
        user_msg = {"role": "user", "content": user_input}

        resp = chat(model=self.model, messages=system_msgs + [user_msg])
        reply = self.sanitize_response(resp["message"]["content"])
        self.history.append(("assistant", reply))
        return reply


    def save_history(self, filename="conversation_history.txt"):
        with open(filename, "w", encoding="utf-8") as f:
            for role, message in self.history:
                f.write(f"{role.title()}: {message}\n\n")

if __name__ == "__main__":
    assistant = NutritionAssistant("gemma_nutri_v4:latest")
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
