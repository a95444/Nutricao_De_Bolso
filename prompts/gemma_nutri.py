import re
from ollama import chat

class NutritionAssistant:
    def __init__(self, model_name):
        self.model = model_name
        # Define profile structure
        self.profile = {
            "Nome": None,
            "Idade": None,
            "Género": None,
            "Altura": None,
            "Peso": None,
            "Nível de Atividade Física": None,
            "Alergias": None,
            "Intolerâncias": None,
            "Condições Médicas": None,
            "Preferências Dietéticas": None,
            "Objetivos de Saúde": None
        }
        self.history = []  # Conversation history
        # Ordered list of profile fields to collect
        self.fields_order = list(self.profile.keys())
        self.current_field_index = 0

    def next_field(self):
        """Return the next unset profile field, or None if all set."""
        while self.current_field_index < len(self.fields_order):
            field = self.fields_order[self.current_field_index]
            if not self.profile[field]:
                return field
            self.current_field_index += 1
        return None

    def update_profile_field(self, field, value):
        """Set a specific profile field."""
        self.profile[field] = value.strip()

    def profile_block(self):
        """Serialize profile for context injection."""
        lines = ["Perfil do Utilizador:"]
        for k, v in self.profile.items():
            if v:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines) + "\n\n" if any(self.profile.values()) else ""

    def sanitize_response(self, text):
        """Ensure disclaimer and domain limitation."""
        disclaimer = "\n\n**Isto não substitui uma consulta com um nutricionista ou profissional de saúde. Para orientações clínicas específicas, consulte um especialista.**"
        if disclaimer not in text:
            text += disclaimer
        forbidden = ["alcoólico", "cocktail", "prescrição farmacológica"]
        if any(word in text.lower() for word in forbidden):
            return "Desculpe, o meu foco restringe-se apenas em nutrição e hábitos saudáveis." + disclaimer
        return text

    def ask(self, user_input):
        # Check if still collecting profile
        field = self.next_field()
        if field:
            # Collect this field
            self.update_profile_field(field, user_input)
            self.history.append(("user", user_input))
            # Move index to next for subsequent calls
            self.current_field_index += 1
            # Ask for next field
            next_f = self.next_field()
            if next_f:
                reply = f"Por favor, indica o teu(a) **{next_f}**:"
            else:
                reply = "Obrigado! Agora que tenho o teu perfil, em que posso ajudar-te em nutrição?"
            self.history.append(("assistant", reply))
            return reply

        # All profile collected: process normal conversation
        # Build messages with profile context
        system_ctx = self.profile_block()
        messages = [{"role": "system", "content": system_ctx},
                    {"role": "user", "content": user_input}]
        # Call the model
        response = chat(model=self.model, messages=messages)
        reply = response['message']['content']
        # Sanitize and log
        reply = self.sanitize_response(reply)
        self.history.append(("user", user_input))
        self.history.append(("assistant", reply))
        return reply

    def save_history(self, filename="conversation_history.txt"):
        """Save conversation history."""
        with open(filename, "w", encoding="utf-8") as f:
            for role, message in self.history:
                f.write(f"{role.title()}: {message}\n\n")

if __name__ == "__main__":
    assistant = NutritionAssistant("nutri-assistant-v2:latest")
    print("Olá! Sou o teu Assistente de Nutrição. Escreve 'sair' para terminar.")
    # Prompt for the first field
    first_field = assistant.next_field()
    print(f"Assistente: Por favor, indica o teu(a) **{first_field}**:")

    while True:
        user_input = input("Tu: ").strip()
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Assistente: Bom apetite e até breve!")
            assistant.save_history()
            break
        reply = assistant.ask(user_input)
        print(f"Assistente: {reply}\n")

