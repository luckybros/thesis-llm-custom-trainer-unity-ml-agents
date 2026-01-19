import re

class ActionParser:

    def __init__(self, settings):
        self.num_agents = settings.num_agents
        self.actions = settings.actions

    def parse_actions(self, text) -> dict:
        result = {}
        matches = re.findall(r'Agent\s+(\d+):(.*?)(?=Agent\s+\d+:|$)', text, re.DOTALL)
        if len(matches) > self.num_agents:
            matches = matches[self.num_agents:]
        print(f"Matches: {matches}")

        for idx_str, acts_str in matches:
            idx = int(idx_str)
            result[idx] = {"continuous" : {}, "discrete": {}}

            for act_name in self.actions["continuous"].keys():
                pattern = re.escape(act_name) + r'\s*\n\s*(.+)'
                match = re.search(pattern, acts_str)

                if match:
                    choice = match.group(1).strip()
                    choice = choice.capitalize()
                    result[idx]["continuous"][act_name] = choice

            for act_name in self.actions["discrete"].keys():
                pattern = re.escape(act_name) + r'\s*\n\s*(.+)'
                match = re.search(pattern, acts_str)

                if match:
                    choice = match.group(1).strip()
                    choice = choice.capitalize()
                    result[idx]["discrete"][act_name] = choice

        return result