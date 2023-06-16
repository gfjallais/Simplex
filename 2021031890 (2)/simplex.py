import numpy as np
import re
import sys

class Simplex:

    def __init__(self):
        self.tableu = np.array([])
        self.unbounded = False
        self.infeasible = False
        self.certificate = []
        self.is_max = True
        self.obj = ""
        self.solution = []
        self.variable_set = {}
        self.aux_variable_set = {}
        self.obj_fun_vars = []

    def build_tableu(self, coeffs, bs, flag):
        self.coeffs = np.vstack(coeffs)
        self.bs = bs
        bs = np.hstack(([0], bs))
        bs = bs.reshape(len(bs), 1)
        identity = np.eye(len(bs) - 1)
        line_of_zeros = np.zeros(len(identity))
        if flag:
            dual_side = np.vstack((line_of_zeros, identity))
            self.tableu = np.hstack((dual_side, self.coeffs, bs))
        else:
            self.tableu = np.hstack((self.coeffs, bs))
        
    def define_entry_base_col(self):
        col_pivot = min(self.tableu[0, 0:-1])
        temp = list(self.tableu[0])
        return temp.index(col_pivot)

    def define_exit_line(self, entry):
        res = {}
        for line in range(1, len(self.tableu)):
            if self.tableu[line][entry] > 0:
                b = self.tableu[line][-1]
                div = b / self.tableu[line][entry]
                res[line] = div
        if len(res) == 0:
            self.certificate = np.zeros(len(self.tableu[0]) - 1 - len(self.bs))
            for idx in range(2, len(self.tableu[0]) - 1):
                pos = self.tableu[0][idx] == 0
                if pos:
                    list_col = list(self.tableu[:, idx])
                    if 1 in list_col:
                        self.certificate[idx-2] = -1 * self.tableu[list_col.index(1), entry]
            self.certificate[entry-2] = 1
            self.unbounded = True
            return -1
        return min(res, key=res.get)

    def calc_new_line(self, line, entry, pivot_line):
        pivot = line[entry] * -1
        res = [val * pivot for val in pivot_line]
        new_line = []
        for i in range(len(res)):
            sum_value = res[i] + line[i]
            new_line.append(sum_value)

        return new_line
    
    def define_pivot_line(self, entry, exit_line):
        line = self.tableu[exit_line]
        pivot = line[entry]
        pivot_line = [val / pivot for val in line]
        return pivot_line
    
    def is_negative(self):
        negative = list(filter(lambda x: x < 0, self.tableu[0, 0:-1]))
        return len(negative) > 0
    
    def calc(self):
        entry = self.define_entry_base_col()
        exit_line = self.define_exit_line(entry)
        if exit_line == -1:
            return
        pivot_line = self.define_pivot_line(entry, exit_line)
        self.tableu[exit_line] = pivot_line

        tableu_copy = self.tableu.copy()

        for idx in range(len(self.tableu)):
            if idx != exit_line:
                line = tableu_copy[idx]
                new_line = self.calc_new_line(line, entry, pivot_line)
                self.tableu[idx] = new_line
    
    def solve(self):
        self.calc()
        while self.is_negative() and not self.unbounded and not self.infeasible:
            self.calc()
        

    def is_infeasible(self):
                        
        tableu_copy = self.tableu[:, len(self.bs):].copy()
        tableu_copy = tableu_copy[:, :-1]
        identity = np.eye(len(self.bs))
        line_of_ones = np.ones(len(identity))
        two_phase_side = np.vstack((line_of_ones, identity))
        tableu_copy[0] = np.zeros(tableu_copy.shape[1])
        tableu_copy = np.hstack((tableu_copy, two_phase_side))
        aux_simplex = Simplex()
        aux_simplex.build_tableu(tableu_copy, self.bs, False)
        
        for i in range(aux_simplex.tableu.shape[1]):
            aux_simplex.tableu[0, i] -= sum(aux_simplex.tableu[1:, i])
        aux_simplex.solve()
        if round(aux_simplex.tableu[0][-1]) < 0:
            self.infeasible = True
        if self.infeasible:
            self.certificate = aux_simplex.tableu[0][-len(self.bs)-1:-1]
        return self.infeasible


    def show_tableu(self):
        for i in range(len(self.tableu)):
            for j in range(len(self.tableu[0])):
               print("{:.2f}".format(self.tableu[i][j]) + "\t", end="")
            print()
        print("-"*60)
        print("Optimal value: ", self.tableu[0][-1] * (1 if self.is_max else -1))


    def get_solution(self):
        if not self.unbounded and not self.infeasible:
            self.certificate = -1 * self.tableu[0][0:len(self.bs)]
        solution = "Status: "
        for var in list(self.variable_set.keys()):
                if var[0:3] != 'aux' and var in self.obj_fun_vars:
                    self.solution.append(str(self.get_value_by_var_name(var)))
        if all(float(v) < 0 for v in self.solution):
            self.infeasible = True
        if self.unbounded:
            solution += "ilimitado\n"
        elif self.infeasible:
            solution += "inviavel\n"
        else:
            solution += "otimo\n"
            solution += "Objetivo: " + str(self.tableu[0][-1] * (1 if self.is_max else -1)) + "\n"
            solution += "Solucao:\n"
            solution += " ".join(self.solution) + "\n"
        solution += "Certificado:\n"
        solution += " ".join([str(v) for v in self.certificate])
        return solution
    
    def get_value_by_var_name(self, string):
        idx = (self.tableu.shape[0] - 1)
        if string in self.variable_set:
            set = self.variable_set
        elif string in self.aux_variable_set:
            set = self.aux_variable_set
            idx += len(self.variable_set)
        set_list = list(set.keys())
        idx += set_list.index(string)
        pos = self.tableu[0][idx] == 0
        if pos:
            list_col = list(self.tableu[:, idx])
            if 1 in list_col:
                if set[string][0] == "keep":
                    return pos * self.tableu[list_col.index(1)][-1]
                elif set[string][0] == "mul":
                    return pos * -1 * self.tableu[np.where(list_col.index(1))[0]][-1]
                elif set[string][0] == "sub":
                    return pos * (self.get_value_by_var_name(set[string][1]) - set[string][2])
                elif set[string][0] == "sub2":
                   return pos * (self.get_value_by_var_name(set[string][1]) - self.get_value_by_var_name(set[string][2]))
        return 0

class Parser:

    operations = ["+", "-", "*", "/"]
    variable_w_coeff_pattern = r"([+-]?\s*(?:\((?:[^)]+)\))?(?:\d[.,]?\d)?(?:\d[\/]?\d)?\d*\s*\*?\s*[a-zA-Z]\d*)"
    variable_pattern = r"[a-zA-Z]\d?"
    constraint_symbols_pattern = r"([<>=]=?)\s*(\d*[+-\/\\*]?\d+)"


    def __init__(self):
        self.variable_set = {}
        self.aux_variable_set = {}
        self.num_aux = 1
        self.list_of_coeffs = []
        self.list_of_b = []
        self.restricted_vars = []
        self.list_of_descounts = []
        self.is_max = True



    def extract_value(self, string):
        string = string.strip()
        if string[0] == "(" and string[-1] == ")":
            return self.extract_value(string[1:-1])
        if string[0] == "-":
            return -1 * self.extract_value(string[1:])
        elif string[0] == "+":
            return self.extract_value(string[1:])
        elif "*" in string:
            num1, num2 = string.split("*")
            return float(num1) * float(num2)
        elif "/" in string:
            numerator, denominator = string.split("/")
            return float(numerator) / float(denominator)
        elif "+" in string:
            num1, num2 = string.split("+")
            return float(num1) + float(num2)
        elif "-" in string:
            num1, num2 = string.split("-")
            return float(num1) - float(num2)
        else:
            return float(string)


    def extract_coefficient(self, string):
        if "*" in string:
            coefficient, variable = string.split("*")
        else:
            variable = string
            coefficient = "1"
        if string[0] == "-":
            variable = string[1:].strip()
            coefficient = str((-1) * self.extract_coefficient(string[1:])[0])
        return self.extract_value(coefficient), variable.strip()   


    def findall_variables(self, line):
        for var in re.findall(self.variable_pattern, line):
            var = var.split()
            name = self.extract_coefficient(var[0])[1]
            if name not in self.variable_set:
                self.variable_set[name] = ("keep", 1)

    def extract_objective_function_state(self, line):
        state = re.findall(self.variable_w_coeff_pattern, line)
        return state
    
    def extract_objective_function_vars(self, line):
        state = re.findall(self.variable_pattern, line)
        return state

    def extract_b_value(self, line):
        constraint_symbols_match = re.findall(self.constraint_symbols_pattern, line)
        return self.extract_value(constraint_symbols_match[0][1])


    def extract_expression_coeff(self, variable_w_coeff_match):
        if len(variable_w_coeff_match) >= 1:        
            count = 0
            array = np.zeros(len(self.variable_set) + len(self.aux_variable_set))
            for var in variable_w_coeff_match:
                var = var.split(" ", 1)
                sub2 = False
                if len(var) > 1:
                    coeff = self.extract_coefficient(var[1])[0]
                    variable = self.extract_coefficient(var[1])[1]
                else:
                    coeff = self.extract_coefficient(var[0])[0]
                    variable = self.extract_coefficient(var[0])[1]

                if self.variable_set[variable][0] == "mul":
                    coeff *= self.variable_set[variable][1]

                elif self.variable_set[variable][0] == "sub":
                    count += coeff * self.variable_set[variable][2]
                    variable = self.variable_set[variable][1]

                elif self.variable_set[variable][0] == "sub2":
                    var1 = self.variable_set[variable][1]
                    var2 = self.variable_set[variable][2]
                    idx1 = list(self.aux_variable_set.keys()).index(var1) + len(self.variable_set)
                    idx2 = list(self.aux_variable_set.keys()).index(var2) + len(self.variable_set)
                    array[idx1] = (coeff)
                    array[idx2] = (-coeff)
                    sub2 = True

                if not sub2:    
                    if variable in self.variable_set:
                        idx = list(self.variable_set.keys()).index(variable)
                    else:
                        idx = list(self.aux_variable_set.keys()).index(variable) + len(self.variable_set)
                    if len(var) > 1:    
                        if(var[0] == "-"):
                            array[idx] = ((-1) * coeff)
                        elif(var[0] == "+"):
                            array[idx] = coeff
                    else:
                        array[idx] = coeff
            self.list_of_descounts.append(count) 
            self.list_of_coeffs.append(array)

    def transform_to_canonical_form(self, line):
        variable_w_coeff_match = re.findall(self.variable_w_coeff_pattern, line)
        constraint_symbols_match = re.findall(self.constraint_symbols_pattern, line)
        if self.extract_b_value(line) != 0:
            self.list_of_b.append(self.extract_b_value(line))
        if len(variable_w_coeff_match) <= 1:
            extracted_c_v = self.extract_coefficient(variable_w_coeff_match[0])
            self.restricted_vars.append(extracted_c_v[1])
            b_temp = float(constraint_symbols_match[0][1]) / extracted_c_v[0]
            if constraint_symbols_match[0][0] == "==":
                self.variable_set[extracted_c_v[1]] = ("sub", b_temp)
            if b_temp == 0:
                if constraint_symbols_match[0][0] == "<=":
                    self.variable_set[extracted_c_v[1]] = ("mul", -1)
        else:
            if constraint_symbols_match[0][0] == "<=":
                variable_w_coeff_match.append("+ aux{num}".format(num = self.num_aux))
                self.variable_set["aux{num}".format(num = self.num_aux)] = ("keep", 1)
                self.num_aux += 1
            elif constraint_symbols_match[0][0] == ">=":
                variable_w_coeff_match.append("- aux{num}".format(num = self.num_aux))
                self.variable_set["aux{num}".format(num = self.num_aux)] = ("keep", 1)
                self.num_aux +=1
        return variable_w_coeff_match

    def subs_non_restricted_var(self):
        for var in self.variable_set.keys():
            if var[0:3] != 'aux':
                if var not in self.restricted_vars:
                    name1 = "aux{num}".format(num = self.num_aux)
                    self.aux_variable_set[name1] = ("keep", 1)
                    self.num_aux +=1
                    name2 = "aux{num}".format(num = self.num_aux)
                    self.aux_variable_set[name2] = ("keep", 1)
                    self.num_aux +=1
                    self.variable_set[var] = ("sub2", name1, name2)

    def correct_bs(self):
        self.list_of_b = [(self.list_of_b[i] - self.list_of_descounts[i]) for i in range(len(self.list_of_b))]

    def correct_fun_obj(self):
        if self.is_max:
            self.list_of_coeffs[0] = [-1 * float(val) for val in self.list_of_coeffs[0]]
    
    def correct_restrictions(self):
        for idx, _ in enumerate(self.list_of_coeffs[1:]):
            op = 1
            if self.list_of_b[idx] < 0:
                op = -1
            self.list_of_coeffs[idx + 1] = [op * val for val in self.list_of_coeffs[idx + 1]]
            self.list_of_b[idx] *= op

simplex = Simplex()
parser = Parser()

with open(sys.argv[1]) as input:

    obj_function_raw = input.readline().split()
    parser.is_max = obj_function_raw[0].upper() == "MAX"
    simplex.is_max = parser.is_max
    
    constraints = input.readlines()
    list_of_states = []
    simplex.obj_fun_vars = parser.extract_objective_function_vars(" ".join(obj_function_raw[1:]))
    parser.findall_variables(" ".join(obj_function_raw[1:]))
    list_of_states.append(parser.extract_objective_function_state(" ".join(obj_function_raw[1:])))

    for line in constraints:
        parser.findall_variables(line)
        list_of_states.append(parser.transform_to_canonical_form(line))
    parser.subs_non_restricted_var()

parser.extract_expression_coeff(list_of_states[0])
for (state, b) in zip(list_of_states[1:], parser.list_of_b):
    parser.extract_expression_coeff(state)

parser.correct_fun_obj()
parser.correct_restrictions()
parser.correct_bs() 

simplex.variable_set = parser.variable_set
simplex.aux_variable_set = parser.aux_variable_set

simplex.build_tableu(parser.list_of_coeffs, parser.list_of_b, True)

if not simplex.is_infeasible():
    simplex.solve()

with open(sys.argv[2], "w+") as output:
    output.write(simplex.get_solution())