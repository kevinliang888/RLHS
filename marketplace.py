import torch
import random
import copy
import itertools
from llm import RestrictTokenLogitsProcessor, set_tokenizer, load_model, llama_chat, get_response, insert_string, append_dict_to_json_file, load_json_file
from prompt import get_all_prompts
from prompt_manager import sample_prices, get_final_feature_prompt, get_ai_option_prompt, get_ai_prompt, get_customer_prompt, generate_purchase_item_prompt, get_final_review_prompt
import pdb


class Marketplace:
    def __init__(self, chatbots=None):
        self.items = {
            "TV": [
                ("Can play 3D movies", "Cannot play 3D movies", "3D movie capability not specified"),
                ("8K Resolution", "4K Resolution", "Resolution not specified"),
                ("HDR", "No HDR", "HDR capability not specified"),
                ("240Hz Refresh Rate", "60Hz Refresh Rate", "Refresh rate not specified"),
                ("Has Smart TV Features", "No Smart TV Features", "Smart features not specified"),
                ("OLED", "LED", "Panel type not specified"),
                ("Built-in Wi-Fi", "Only Ethernet", "Network connectivity not specified"),
                ("Large screen (65 inches or more)", "Small screen (less than 32 inches)", "Screen size not specified"),
            ],
            "Laptop": [
                ("High resolution (4K)", "Standard resolution (1080p)", "Resolution not specified"),
                ("Latest generation processor", "Older generation processor", "Processor generation not specified"),
                ("16GB or more RAM", "8GB or less RAM", "RAM not specified"),
                ("SSD storage", "HDD storage", "Storage type not specified"),
                ("Long battery life (10+ hours)", "Short battery life (less than 5 hours)", "Battery life not specified"),
                ("Lightweight (less than 2 kg)", "Heavy (more than 3 kg)", "Weight not specified"),
                ("Multiple USB-C ports", "Few or no USB-C ports", "USB-C ports not specified"),
                ("Fast charging capability", "No fast charging", "Charging speed not specified")
            ],
            "Smartphone": [
                ("High-resolution camera (108MP)", "Standard camera (12MP)", "Camera resolution not specified"),
                ("Large battery capacity (5000mAh)", "Small battery capacity (3000mAh)", "Battery capacity not specified"),
                ("OLED display", "LCD display", "Display type not specified"),
                ("Large storage (256GB+)", "Small storage (64GB)", "Storage capacity not specified"),
                ("8GB or more RAM", "4GB or less RAM", "RAM not specified"),
                ("Supports 5G", "Does not support 5G", "5G capability not specified"),
                ("Has biometric security (e.g., fingerprint, face recognition)", "No biometric security", "Biometric security not specified"),
                ("Fast charging capability", "No fast charging", "Charging speed not specified")
            ],
            "Refrigerator": [
                ("Large capacity (500+ liters)", "Small capacity (200 liters)", "Capacity not specified"),
                ("Energy-efficient (Energy Star certified)", "Not energy-efficient", "Energy efficiency not specified"),
                ("Frost-free", "Manual defrost", "Defrost type not specified"),
                ("Digital temperature control", "Manual temperature control", "Temperature control not specified"),
                ("Has water dispenser", "No water dispenser", "Water dispenser not specified"),
                ("Built-in ice maker", "No ice maker", "Ice maker not specified"),
                ("Quiet operation", "Noisy operation", "Noise level not specified"),
                ("Adjustable shelves", "Fixed shelves", "Shelving not specified")
            ],
            "Washing Machine": [
                ("Front-loading", "Top-loading", "Loading type not specified"),
                ("Large capacity (10kg+)", "Small capacity (5kg or less)", "Capacity not specified"),
                ("Energy-efficient (Energy Star certified)", "Not energy-efficient", "Energy efficiency not specified"),
                ("Quick wash feature", "No quick wash feature", "Wash feature not specified"),
                ("Built-in dryer", "No built-in dryer", "Dryer inclusion not specified"),
                ("Silent operation", "Noisy operation", "Noise level not specified"),
                ("Smart features (e.g., Wi-Fi, app control)", "No smart features", "Smart features not specified"),
                ("Multiple wash programs", "Limited wash programs", "Wash programs not specified")
            ],
            "Microwave Oven": [
                ("High wattage (1000W+)", "Low wattage (700W or less)", "Wattage not specified"),
                ("Convection feature", "No convection feature", "Convection feature not specified"),
                ("Inverter technology", "No inverter technology", "Inverter technology not specified"),
                ("Sensor cooking", "No sensor cooking", "Sensor technology not specified"),
                ("Large capacity (1.5 cubic feet+)", "Small capacity (0.7 cubic feet or less)", "Capacity not specified"),
                ("Child lock feature", "No child lock feature", "Child lock not specified"),
                ("Quick-start button", "No quick-start button", "Quick-start feature not specified"),
                ("Touch control panel", "Dial control panel", "Control panel not specified")
            ],
            "Dishwasher": [
                ("Built-in", "Portable", "Installation type not specified"),
                ("Energy-efficient (Energy Star certified)", "Not energy-efficient", "Energy efficiency not specified"),
                ("Quiet operation", "Noisy operation", "Noise level not specified"),
                ("Third rack for utensils", "No third rack", "Third rack not specified"),
                ("Adjustable racks", "Non-adjustable racks", "Rack adjustability not specified"),
                ("Multiple wash cycles", "Limited wash cycles", "Wash cycles not specified"),
                ("Delay start feature", "No delay start feature", "Start feature not specified"),
                ("Soil sensor", "No soil sensor", "Soil sensor not specified")
            ],
            "Air Conditioner": [
                ("High BTU (12,000+)", "Low BTU (5,000 or less)", "BTU rating not specified"),
                ("Energy-efficient (Energy Star certified)", "Not energy-efficient", "Energy efficiency not specified"),
                ("Inverter technology", "No inverter technology", "Inverter technology not specified"),
                ("Smart features (e.g., Wi-Fi, app control)", "No smart features", "Smart features not specified"),
                ("Quiet operation", "Noisy operation", "Noise level not specified"),
                ("Programmable timer", "No programmable timer", "Timer not specified"),
                ("Remote control", "No remote control", "Remote control not specified"),
                ("Washable filter", "Disposable filter", "Filter type not specified")
            ],
            "Vacuum Cleaner": [
                ("Cordless", "Corded", "Power type not specified"),
                ("Bagless", "Bagged", "Bag type not specified"),
                ("HEPA filter", "No HEPA filter", "Filter type not specified"),
                ("Lightweight (less than 3kg)", "Heavy (more than 5kg)", "Weight not specified"),
                ("Long battery life (60+ minutes)", "Short battery life (20 minutes or less)", "Battery life not specified"),
                ("Quiet operation", "Noisy operation", "Noise level not specified"),
                ("Smart features (e.g., Wi-Fi, app control)", "No smart features", "Smart features not specified"),
                ("Large dustbin capacity", "Small dustbin capacity", "Dustbin capacity not specified")
            ],
            "Coffee Maker": [
                ("Programmable settings", "Manual settings", "Programmability not specified"),
                ("Built-in grinder", "No built-in grinder", "Grinder inclusion not specified"),
                ("Multiple brew strengths", "Single brew strength", "Brew strength not specified"),
                ("Thermal carafe", "Glass carafe", "Carafe type not specified"),
                ("Large capacity (12 cups+)", "Small capacity (1-4 cups)", "Capacity not specified"),
                ("Energy-efficient", "Not energy-efficient", "Energy efficiency not specified"),
                ("Water filtration system", "No water filtration system", "Water filtration not specified"),
                ("Compact design", "Bulky design", "Design size not specified")
            ]
        }

        self.prices = {
            "TV": [(1800, 1900), (1400, 1600), (900, 1100)],
            "Laptop": [(1500, 1600), (1200, 1400), (800, 1000)],
            "Smartphone": [(1000, 1100), (800, 900), (600, 700)],
            "Refrigerator": [(2200, 2400), (1800, 2000), (1400, 1600)],
            "Washing Machine": [(1500, 1600), (1200, 1400), (900, 1100)],
            "Microwave Oven": [(550, 600), (450, 500), (350, 400)],
            "Dishwasher": [(900, 1000), (700, 800), (500, 600)],
            "Air Conditioner": [(2100, 2200), (1800, 2000), (1500, 1700)],
            "Vacuum Cleaner": [(350, 400), (300, 350), (250, 300)],
            "Coffee Maker": [(350, 400), (300, 350), (250, 300)]
        }

        self.init_prompt()
        self.all_cases = self.init_all_cases()
        self.chatbots = chatbots

    def init_all_cases(self):
        # values = [-1, 0]
        values = [-1, 0, 1]
        neg_cases = list(itertools.product(values, repeat=3))
        price_req_cases = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)]
        no_price_cases = [(1, 1, -1), (1, 1, 0), (1, 1, 1)]
        all_cases = {"neg_cases": neg_cases, "price_req_cases": price_req_cases, "no_price_cases": no_price_cases}
        return all_cases

    def init_prompt(self):
        all_prompts = get_all_prompts()
        self.pref_prompt = all_prompts["pref_prompt"]
        self.final_pref_prompt = all_prompts["final_pref_prompt"]
        self.decision_pref_prompt = all_prompts["decision_pref_prompt"]
        self.final_decision_pref_prompt = all_prompts["final_decision_pref_prompt"]
        self.purchase_review_prompt = all_prompts["purchase_review_prompt"]
        self.no_purchase_review_prompt = all_prompts["no_purchase_review_prompt"]
        self.feedback = all_prompts["feedback_prompt"]

        self.human_pref_prompt_template = all_prompts["human_pref_prompt_template"]
        self.human_prompt_template = all_prompts["human_prompt_template"]
        self.price_prompt = all_prompts["price_prompt"]
        self.ai_prompt_template = all_prompts["ai_prompt_template"]
        self.item_choice_prompt = all_prompts["item_choice_prompt"]
        self.final_item_choice_prompt = all_prompts["final_item_choice_prompt"]
        self.final_review_prompt_template = all_prompts["final_review_prompt_template"]
        self.satisfaction_prompt = all_prompts["satisfaction_prompt"]
        self.final_satisfaction_prompt = all_prompts["final_satisfaction_prompt"]

    def true_reward(self, case, price_rank, lowest_price_req):
        """ True reward will be adjusted later according to human actions. """
        if case != 1:
            return -1

        if not lowest_price_req:
            return 1

        if price_rank == 1:
            return 0
        elif price_rank == 2:
            return 1
        else:
            return 2

    def generate_scenarios(self):
        option_lists = ['A', 'B', 'C']
        all_results = []
        for item in self.items:
            all_features = self.items[item]
            price_ranges = self.prices[item]
            for j in range(len(all_features)):
                feature_tuple = all_features[j]
                for case_name in self.all_cases:
                    cases = self.all_cases[case_name]
                    for case in cases:
                        human_prompt_template_new = self.human_prompt_template
                        human_pref_prompt_template_new = self.human_pref_prompt_template
                        lowest_price_req = False
                        if case_name == "price_req_cases" or case_name == "no_price_cases":
                            lowest_price_req = True
                            human_prompt_template_new = self.human_prompt_template + " " + self.price_prompt
                            human_pref_prompt_template_new = self.human_pref_prompt_template + " " + self.price_prompt

                        all_prices = sample_prices(price_ranges)
                        random.shuffle(option_lists)
                        all_info_dict = {}
                        for i in range(len(all_prices)):
                            if case[i] == -1:
                                feature = feature_tuple[1]
                            elif case[i] == 1:
                                feature = feature_tuple[0]
                            else:
                                feature = feature_tuple[2]
                            other_features = [random.choice(all_features[k][:2]) for k in range(len(all_features)) if k != j]
                            lowest_price = True if i == 2 else False
                            price_rank = i + 1
                            reward = self.true_reward(case[i], price_rank, lowest_price_req)
                            is_price_aval = not (case_name == "no_price_cases")
                            dict_example = {
                                "price": all_prices[i],
                                "feature": feature,
                                "other_features": other_features,
                                "case": case[i],
                                "price_rank": price_rank,
                                "req_feature": feature_tuple[0],
                                "bad_feature": feature_tuple[1],
                                "reward": reward,
                                "is_price_aval": is_price_aval
                            }
                            all_info_dict[option_lists[i]] = dict_example

                        ai_prompt = get_ai_prompt(self.ai_prompt_template, item, all_info_dict)
                        customer_prompt = get_customer_prompt(human_prompt_template_new, item, all_info_dict, feature_tuple[0])
                        customer_pref_prompt = get_customer_prompt(human_pref_prompt_template_new, item, all_info_dict, feature_tuple[0])
                        human_review_prompt = get_final_review_prompt(self.final_review_prompt_template, all_info_dict)

                        results = {}
                        results["option_info"] = all_info_dict
                        results["case_name"] = case_name
                        results["all_cases"] = case
                        results["lowest_price_req"] = lowest_price_req
                        results["item"] = item

                        results["ai_prompt"] = ai_prompt
                        results["human_prompt"] = customer_prompt
                        results["human_pref_prompt"] = customer_pref_prompt
                        results["human_review_prompt"] = human_review_prompt

                        all_results.append(results)

        return all_results

    def find_first_available_string(self, results, format_str):
        num = 1
        while True:
            key = format_str.format(num)
            if key not in results:
                return key
            num += 1


    def get_sample_dict(self, ai_response, ai_chat, customer_chat, choice=None):
        sample_dict = {}
        sample_dict["response"] = ai_response
        sample_dict["ai_chat"] = ai_chat
        sample_dict["customer_chat"] = customer_chat
        sample_dict["choice"] = choice
        return sample_dict


    def generate_new(self, all_results, output_path=None, ckpts_list=None, rlhf_type=['base', 'phs', 'fhs', 'phs_v0', 'fhs_v0']):
        '''
        This is for asking questions in human study only.
        '''
        # rlhf_type: "all", "base", "phs", "fhs"
        iters = 0
        init_chatbot = self.chatbots[0]
        options = ["A", "B", "C"]

        for results in all_results:
            iters += 1
            all_cases = results["all_cases"]
            info = results["option_info"]
            print("#######################")
            print(f"Iters: {iters}, {results['item']}: {info['A']['req_feature']}, Case name: {results['case_name']}, Cases: {all_cases}")
            ai_prompt = results['ai_prompt'] 
            customer_prompt = results['human_prompt'] 
            human_pref_chat = results["human_pref_prompt"]
            human_review_prompt = results["human_review_prompt"]
            case_name = results["case_name"]
            if "ai_chat" in results and "human_chat" in results:
                init_ai_chat, init_customer_chat = results["ai_chat"], results["human_chat"]

                # only ask for feature
                item = results["item"]
                req_feature = results["option_info"]["A"]["req_feature"]

                # if results['case_name'] == "no_price_cases":
                #     init_ai_chat[1]['content'] = f"Could you please tell me which {item} includes the feature: {req_feature}? I am also interested in their prices."
                #     init_customer_chat[2]['content'] = f"Could you please tell me which {item} includes the feature: {req_feature}? I am also interested in their prices."
                # else:

                    # init_ai_chat[1]['content'] = f"Could you please tell me which {item} includes the feature: {req_feature}?"
                    # init_customer_chat[2]['content'] = f"Could you please tell me which {item} includes the feature: {req_feature}?"

                # init_ai_chat[1]['content'] = f"Could you please tell me which {item} includes the feature: {req_feature}?"
                # init_customer_chat[2]['content'] = f"Could you please tell me which {item} includes the feature: {req_feature}?"

                # word = 'an' if item == "Air Conditioner" else 'a'
                # init_ai_chat[1]['content'] = f"I'm looking for {word} {item} that includes the feature: {req_feature}. Can you help me?"
                # init_customer_chat[2]['content'] = f"I'm looking for {word} {item} that includes the feature: {req_feature}. Can you help me?"

                init_ai_chat[1]['content'] = f"Could you please tell me only the price of each {item}?"
                init_customer_chat[2]['content'] = f"Could you please tell me only the price of each {item}?"


                # simplify the system prompt
                prompt = "You should keep responses concise, and limited to two paragraphs."
                content = init_ai_chat[0]['content']
                new = content.split("\n\n")
                new[0] += " " + prompt
                init_ai_chat[0]['content'] = "\n\n".join(new)
            else:
                init_ai_chat, init_customer_chat = init_chatbot.get_initial_chat(ai_prompt, customer_prompt)
                results["ai_chat"] = init_ai_chat
                results["human_chat"] = init_customer_chat

            total_iter = 0
            for i in range(len(self.chatbots)):
                chatbot = self.chatbots[i]

                # Check if we need to recompute the dictionary
                create_new_dict = False
                model_name = chatbot.model_name
                model_results = results.get(model_name)
                if model_results is None:
                    results[model_name] = {}
                    create_new_dict = True
                elif ckpts_list:
                    sample_dict = model_results.get(ckpts_list[i])
                    create_new_dict = sample_dict is None
                else:
                    sample_dict = model_results
                
                if create_new_dict:
                    response, ai_chat, customer_chat = chatbot.get_ai_response2(init_customer_chat, init_ai_chat)
                    ai_chat[-2]['content'] = f'Could you please tell me the price of each {item}?'
                    customer_chat[-2]['content'] = f'Could you please tell me the price of each {item}?'

                    word = 'an' if item == "Air Conditioner" else 'a'
                    question = f"I'm looking for {word} {item} that includes the feature: {req_feature}. Can you help me?"

                    ai_chat.append({'role': 'user', 'content': question})
                    customer_chat.append({'role': 'assistant', 'content': question})
                    response, ai_chat, customer_chat = chatbot.get_ai_response2(customer_chat, ai_chat)

                    sample_dict = self.get_sample_dict(None, ai_chat, customer_chat, choice=None)
                    if ckpts_list:
                        results[model_name][ckpts_list[i]] = sample_dict


            if output_path:
                append_dict_to_json_file(results, output_path)
        return all_results


    def generate_test_data(self, all_results, output_path=None, ckpts_list=None, rlhf_type=['base', 'phs', 'fhs', 'phs_v0', 'fhs_v0']):
        '''
        This is for test time inference.
        '''
        # rlhf_type: "all", "base", "phs", "fhs"
        iters = 0
        init_chatbot = self.chatbots[0]
        options = ["A", "B", "C"]

        for results in all_results:
            iters += 1

            all_cases = results["all_cases"]
            info = results["option_info"]
            print("#######################")
            print(f"Iters: {iters}, {results['item']}: {info['A']['req_feature']}, Case name: {results['case_name']}, Cases: {all_cases}")
            ai_prompt = results['ai_prompt'] 
            customer_prompt = results['human_prompt'] 
            human_pref_chat = results["human_pref_prompt"]
            human_review_prompt = results["human_review_prompt"]
            case_name = results["case_name"]
            if "ai_chat" in results and "human_chat" in results:
                init_ai_chat, init_customer_chat = results["ai_chat"], results["human_chat"]
            else:
                init_ai_chat, init_customer_chat = init_chatbot.get_initial_chat(ai_prompt, customer_prompt)
                results["ai_chat"] = init_ai_chat
                results["human_chat"] = init_customer_chat

            total_iter = 0
            for i in range(len(self.chatbots)):
                chatbot = self.chatbots[i]

                # Check if we need to recompute the dictionary
                create_new_dict = False
                model_name = chatbot.model_name
                model_results = results.get(model_name)
                if model_results is None:
                    results[model_name] = {}
                    create_new_dict = True
                elif ckpts_list:
                    sample_dict = model_results.get(ckpts_list[i])
                    create_new_dict = sample_dict is None
                else:
                    sample_dict = model_results
                
                if create_new_dict:
                    choice, ai_chat, customer_chat = chatbot.get_further_chat(init_ai_chat, init_customer_chat)
                    ai_response = "\n\n".join(ai_chat[-2]['content'].split("\n\n")[:-1])
                    sample_dict = self.get_sample_dict(ai_response, ai_chat, customer_chat, choice)
                    if ckpts_list:
                        results[model_name][ckpts_list[i]] = sample_dict

                # Use for RLHF or RLHS
                choice = sample_dict["choice"]
                customer_chat = sample_dict["customer_chat"]

                if "base" in rlhf_type:
                    reason, rating = chatbot.get_satisfaction(customer_chat)
                    sample_dict["initial_reason"] = reason
                    sample_dict["initial_rating"] = rating

                if "phs" in rlhf_type:
                    # Partial Hindsight simulation enabled
                    customer_extend_chat = chatbot.update_customer_state_partial(customer_chat, info, choice)
                    hs_reason, hs_rating = chatbot.get_satisfaction(customer_extend_chat)
                    sample_dict["phs_reason"] = hs_reason
                    sample_dict["phs_rating"] = hs_rating
                    
                if "fhs" in rlhf_type:
                    # Full Hindsight simulation enabled
                    customer_extend_chat_full = chatbot.update_customer_state_full(customer_chat, info, choice, human_review_prompt)
                    oracle_hs_reason, oracle_hs_rating = chatbot.get_satisfaction(customer_extend_chat_full)
                    sample_dict["fhs_reason"] = oracle_hs_reason
                    sample_dict["fhs_rating"] = oracle_hs_rating

                if "phs_v0" in rlhf_type:
                    # Partial Hindsight simulation (only decision) enabled
                    instruction = [customer_chat[0]]
                    customer_extend_chat = chatbot.update_customer_state_partial(instruction, info, choice)
                    hs_v0_reason, hs_v0_rating = chatbot.get_decision_satisfaction(customer_extend_chat)
                    sample_dict["phs_v0_reason"] = hs_v0_reason
                    sample_dict["phs_v0_rating"] = hs_v0_rating

                if "fhs_v0" in rlhf_type:
                    # Full Hindsight simulation (only decision) enabled
                    instruction = [customer_chat[0]]
                    customer_extend_chat_full = chatbot.update_customer_state_full(instruction, info, choice, human_review_prompt)
                    oracle_hs_v0_reason, oracle_hs_v0_rating = chatbot.get_decision_satisfaction(customer_extend_chat_full)
                    sample_dict["fhs_v0_reason"] = oracle_hs_v0_reason
                    sample_dict["fhs_v0_rating"] = oracle_hs_v0_rating

            if output_path:
                append_dict_to_json_file(results, output_path)
        return all_results


    def generate_rlhf_data(self, all_results, output_path=None, final_output_path=None, thres=15):
        '''
        This is for generating RLHF data.
        '''
        options = ["A", "B", "C"]
        chatbot = self.chatbots[0] if isinstance(self.chatbots, list) else self.chatbots
        # load the last running checkpoint
        data = load_json_file(output_path)
        all_results[:len(data)] = data
        iters = len(data)

        for results in all_results[len(data):]:
            iters += 1
            all_cases = results["all_cases"]
            info = results["option_info"]
            print("#######################")
            print(f"Iters: {iters}, {results['item']}: {info['A']['req_feature']}, Case name: {results['case_name']}, Cases: {all_cases}")

            ai_prompt = results['ai_prompt'] 
            customer_prompt = results['human_prompt'] 
            human_pref_chat = results["human_pref_prompt"]
            human_review_prompt = results["human_review_prompt"]
            case_name = results["case_name"]
            init_ai_chat, init_customer_chat = chatbot.get_initial_chat(ai_prompt, customer_prompt)
            init_ai_chat1, init_ai_chat2 = init_ai_chat, copy.deepcopy(init_ai_chat)
            
            dialog = chatbot.construct_two_dialogs(human_pref_chat, customer_chat1, customer_chat2)
            reason, choice = chatbot.get_final_pref(dialog)
                
            if choice == "1":
                chosen_chat = ai_chat1
                reject_chat = ai_chat2
            if choice == "2":
                chosen_chat = ai_chat2
                reject_chat = ai_chat1
            
            chosen = "\n\n".join(chosen_chat[-2]['content'].split("\n\n")[:-1])
            reject = "\n\n".join(reject_chat[-2]['content'].split("\n\n")[:-1])
            
            
            # RLHF
            results["dialog"] = ai_chat1[:2]
            results["chosen"] = chosen
            results["reject"] = reject
            
            # Log AI and customer responses
            results["ai_chat_1"] = ai_chat1
            results["ai_chat_2"] = ai_chat2
            results["human_chat_1"] = customer_chat1
            results["human_chat_2"] = customer_chat2
            results["choice_1"] = choice1
            results["choice_2"] = choice2
            
            # Preference
            results["pref_eval_prompt"] = dialog
            results["pref_reason"] = reason
            results["pref_choice"] = choice
            
            
            # partial hindsight
            customer_extend_chat1 = chatbot.update_customer_state_partial(customer_chat1, info, choice1)
            customer_extend_chat2 = chatbot.update_customer_state_partial(customer_chat2, info, choice2)
            dialog2 = chatbot.construct_two_dialogs(human_pref_chat, customer_extend_chat1, customer_extend_chat2)
            final_reason2, final_choice2 = chatbot.get_final_pref(dialog2) 
            part_state_1, part_state_2 = customer_extend_chat1[-1]["content"], customer_extend_chat2[-1]["content"]
            results["partial_hindsight_eval_prompt"] = dialog2
            results["partial_hindsight_reason"] = final_reason2
            results["partial_hindsight_choice"] = final_choice2
            results["partial_hindsight_state_1"] = part_state_1
            results["partial_hindsight_state_2"] = part_state_2
            
            # partial hindsight v0
            dialog2_v0 = chatbot.construct_two_decisions(human_pref_chat, part_state_1, part_state_2)
            final_reason2_v0, final_choice2_v0 = chatbot.get_final_pref(dialog2_v0, "decision") 
            results["partial_hindsight_eval_prompt_v0"] = dialog2_v0
            results["partial_hindsight_reason_v0"] = final_reason2_v0
            results["partial_hindsight_choice_v0"] = final_choice2_v0
            
            # Full hindsight
            customer_full_chat1 = chatbot.update_customer_state_full(customer_chat1, info, choice1, human_review_prompt)
            customer_full_chat2 = chatbot.update_customer_state_full(customer_chat2, info, choice2, human_review_prompt)
            dialog3 = chatbot.construct_two_dialogs(human_pref_chat, customer_full_chat1, customer_full_chat2)
            final_reason3, final_choice3 = chatbot.get_final_pref(dialog3)
            full_state_1, full_state_2 = customer_full_chat1[-1]["content"], customer_full_chat2[-1]["content"]
            results["full_hindsight_eval_prompt"] = dialog3
            results["full_hindsight_reason"] = final_reason3
            results["full_hindsight_choice"] = final_choice3
            results["full_hindsight_state_1"] = full_state_1
            results["full_hindsight_state_2"] = full_state_2
            
            # Full hindsight v0
            dialog3_v0 = chatbot.construct_two_decisions(human_pref_chat, full_state_1, full_state_2)
            final_reason3_v0, final_choice3_v0 = chatbot.get_final_pref(dialog3_v0, "decision") 
            results["full_hindsight_eval_prompt_v0"] = dialog3_v0
            results["full_hindsight_reason_v0"] = final_reason3_v0
            results["full_hindsight_choice_v0"] = final_choice3_v0

            if output_path:
                append_dict_to_json_file(results, output_path)
        
        return all_results