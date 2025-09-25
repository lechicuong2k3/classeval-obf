import ast
import keyword
import hashlib
from typing import Dict, Set, Optional, Tuple, Sequence, List

# Compact default dictionary of medical-ish words for meaningful obfuscation (scheme 4)
_DEFAULT_MEDICAL_WORDS: Tuple[str, ...] = (
    "aorta","atrium","ventricle","alveolus","bronchiole","trachea","larynx","pharynx","esophagus","duodenum",
    "jejunum","ileum","colon","rectum","pancreas","hepatocyte","bile","nephron","glomerulus","ureter",
    "urethra","cortex","medulla","hypothalamus","pituitary","thyroid","parathyroid","adrenal","gonad","synapse",
    "axon","dendrite","myelin","neuron","glia","astrocyte","microglia","oligodendrocyte","sarcomere","myofibril",
    "tendon","ligament","cartilage","periosteum","osteocyte","osteoclast","osteoblast","hemoglobin","erythrocyte",
    "leukocyte","thrombocyte","plasma","lymph","lymphocyte","antigen","antibody","cytokine","histamine","insulin",
    "glucagon","cortisol","adrenaline","serotonin","dopamine","acetylcholine","gaba","glutamate","synovial","meniscus"
)

def _collect_defined_names(tree: ast.AST) -> Set[str]:
    defined: Set[str] = set()
    def add_target(t: ast.AST):
        if isinstance(t, ast.Name):
            defined.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for a in list(getattr(args, 'posonlyargs', [])) + list(args.args) + list(getattr(args, 'kwonlyargs', [])):
                    defined.add(a.arg)
                if args.vararg:
                    defined.add(args.vararg.arg)
                if args.kwarg:
                    defined.add(args.kwarg.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                add_target(t)
        elif isinstance(node, ast.AnnAssign):
            add_target(node.target)
        elif isinstance(node, ast.AugAssign):
            add_target(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            add_target(node.target)
        elif isinstance(node, ast.With):
            for i in node.items:
                if i.optional_vars:
                    add_target(i.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                defined.add(n)
        elif isinstance(node, ast.Lambda):
            args = node.args
            for a in list(getattr(args, 'posonlyargs', [])) + list(args.args) + list(getattr(args, 'kwonlyargs', [])):
                defined.add(a.arg)
            if args.vararg:
                defined.add(args.vararg.arg)
            if args.kwarg:
                defined.add(args.kwarg.arg)
        elif isinstance(node, ast.comprehension):
            add_target(node.target)
    return defined

class _Renamer(ast.NodeTransformer):

    def __init__(self, seed: str = "obf", *, scheme: int = 1, name_length: str = "long", dictionary_words: Optional[Sequence[str]] = None, extra_reserved: Optional[Set[str]] = None, pre_imported: Optional[Set[str]] = None, defined_names: Optional[Set[str]] = None):
        self.seed = seed
        self.scheme = scheme or 1
        self.name_length = name_length
        self.map: Dict[str, str] = {}
        self.reserved: Set[str] = set(keyword.kwlist) | set(dir(__import__("builtins")))
        if extra_reserved:
            self.reserved |= set(extra_reserved)
        self.current_scope_stack: list[Set[str]] = []
        self.imported: Set[str] = set(pre_imported or set())
        self._annotation_depth: int = 0
        self.class_names: Set[str] = set()
        self.defined: Set[str] = set(defined_names or set())
        self._call_func_stack: list[str] = []
        self._in_class_context: bool = False  # Track if we're currently inside a class definition
        self._current_class_methods: Set[str] = set()  # Track methods defined in current class
        self._method_renames: Dict[str, str] = {}  # Track method name mappings within current class
        self._variable_renames: Dict[str, str] = {}  # Track all variable renames for consistency
        self._global_method_renames: Dict[str, str] = {}  # Track method renames globally across all classes
        self._global_attribute_renames: Dict[str, str] = {}  # Track instance attribute renames globally
        # Type counters for scheme 12 (type-based names)
        self._type_counters: Dict[str, int] = {"class": 0, "function": 0, "var": 0}
        # Misleading name pools for scheme 13
        self._misleading_function_names: Tuple[str, ...] = (
            "compute_max","sort_desc","is_valid","load_config","encode_password",
            "decrypt","merge_sorted","find_min","is_empty","send_email",
            "train_model","predict_positive","connect_db","flush_cache","write_json"
        )
        self._misleading_variable_names: Tuple[str, ...] = (
            "is_empty","max_value","min_value","total_sum","index_map",
            "error","config_path","is_ready","cache","count_map",
            "avg","threshold","enabled","mutex","queue"
        )
        self._misleading_class_names: Tuple[str, ...] = (
            "DatabaseConnection","HttpServer","JsonParser","LRUCache","ThreadPool",
            "FileLock","ImageDecoder","EventEmitter","Graph","Trie",
            "AuthManager","MessageQueue","RateLimiter","BinaryHeap","BloomFilter"
        )
        # Protected regex-like method names (module and compiled pattern methods)
        self._protected_regex_methods: Set[str] = {
            'match', 'search', 'fullmatch', 'findall', 'finditer', 'split', 'sub', 'subn', 'compile'
        }
        # Prepare dictionary words for scheme 4
        words_source = tuple(dictionary_words) if dictionary_words else _DEFAULT_MEDICAL_WORDS
        filtered = tuple(w for w in words_source if isinstance(w, str) and w.isidentifier() and w not in keyword.kwlist)
        self.dict_words: Tuple[str, ...] = filtered if filtered else _DEFAULT_MEDICAL_WORDS
        # Pool for scheme 10 (single-character identifiers). Must be valid identifier-start characters.
        self._single_char_pool: Tuple[str, ...] = tuple(
            (
                # ASCII letters
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                # Greek lowercase and uppercase
                "αβγδεζηθικλμνξοπρστυφχψω"
                "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
                # Cyrillic lowercase and uppercase (subset of common letters)
                "абвгдеёжзийклмнопрстуфхцчшщыэюя"
                "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯ"
            )
        )

    def _get_length_multiplier(self) -> float:
        """Get length multiplier based on name_length parameter."""
        if self.name_length == "short":
            return 0.4
        elif self.name_length == "medium":
            return 0.7
        else:  # "long" or any other value
            return 1.0

    def _collect_class_method_renames(self, node: ast.ClassDef) -> None:
        """Collect method renames from a class definition without processing the body."""
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                old_name = item.name
                # Do not rename protected regex-like method names
                if old_name in self._protected_regex_methods:
                    continue
                new_name = self._rename(old_name, kind="function")
                self._global_method_renames[old_name] = new_name
                self.map[old_name] = new_name

    def _is_dunder(self, name: str) -> bool:
        return len(name) >= 4 and name.startswith("__") and name.endswith("__")

    def _gen_name(self, name: str, kind: Optional[str] = None) -> str:
        # Multiple deterministic obfuscation strategies. All produce identifiers.
        seed_input = (self.seed + ":" + name).encode()
        hex_digest = hashlib.sha256(seed_input).hexdigest()
        # Strategy 13: Misleading names by type (deterministic from pools)
        if self.scheme == 13:
            if kind == "class":
                pool = self._misleading_class_names
                default_prefix = "Class"
            elif kind == "function":
                pool = self._misleading_function_names
                default_prefix = "func"
            else:
                pool = self._misleading_variable_names
                default_prefix = "var"
            if pool:
                base_idx = int(hex_digest[:8], 16) % len(pool)
                # Try pool name first, then append numeric suffixes if needed
                attempt = 0
                while True:
                    candidate = pool[(base_idx + attempt) % len(pool)]
                    if attempt > 0:
                        candidate = f"{candidate}{attempt}"
                    if candidate not in self.reserved and candidate not in self.imported and candidate not in self.map.values():
                        return candidate
                    attempt += 1
                    if attempt > 100:
                        break
            # Fallback if pool exhausted or empty
            suffix = int(hex_digest[:6], 16) % 10000
            candidate = f"{default_prefix}{suffix}"
            return candidate
        # Strategy 12: Type-based simple counters (class1, function1, var1, ...)
        if self.scheme == 12:
            prefix = "var"
            if kind == "class":
                prefix = "class"
            elif kind == "function":
                prefix = "function"
            # Increment counter until we find an unused name
            while True:
                self._type_counters[prefix] = self._type_counters.get(prefix, 0) + 1
                candidate = f"{prefix}{self._type_counters[prefix]}"
                if candidate not in self.reserved and candidate not in self.imported and candidate not in self.map.values():
                    return candidate


        # Strategy 10: Single-character identifiers from a large valid Unicode pool
        if self.scheme == 10:
            if not self._single_char_pool:
                # Fallback to underscore if pool somehow empty
                return "_"
            start_idx = int(hex_digest[:8], 16) % len(self._single_char_pool)
            used_names = set(self.map.values()) | self.reserved | self.imported
            # Probe for an unused single character
            for i in range(len(self._single_char_pool)):
                candidate = self._single_char_pool[(start_idx + i) % len(self._single_char_pool)]
                # Ensure it's a valid identifier and not already used/reserved/imported
                if candidate.isidentifier() and candidate not in used_names:
                    return candidate
            # As a last resort (pool exhausted), return a deterministic single char even if reused
            return self._single_char_pool[start_idx]

        # Strategy 1 (default): existing visually confusing ASCII with underscores
        if self.scheme == 1:
            trans = str.maketrans({
                "a":"l","b":"I","c":"O","d":"o","e":"0","f":"1",
                "0":"O","1":"l","2":"I","3":"o","4":"0","5":"1",
                "6":"O","7":"l","8":"I","9":"o"
            })
            mapped = hex_digest.translate(trans)
            base_length = 24 + (int(hex_digest[0], 16) % 3) * 8  # 24, 32, or 40
            length = int(base_length * self._get_length_multiplier())
            body = mapped[:length]
            segments = [body[i:i+8] for i in range(0, len(body), 8)]
            return "_" + "_".join(segments)

        # Strategy 2: Ambiguous ASCII using only l/I/1/O/0/o with jittered underscores
        if self.scheme == 2:
            alphabet = "lI1O0o"
            chars: list[str] = []
            for i in range(0, len(hex_digest), 2):
                pair = hex_digest[i:i+2]
                val = int(pair, 16)
                chars.append(alphabet[(val >> 3) % len(alphabet)])
                chars.append(alphabet[(val ^ (val >> 1)) % len(alphabet)])
            base_length = 24 + (int(hex_digest[1], 16) % 3) * 8
            length = int(base_length * self._get_length_multiplier())
            body = "".join(chars)[:length]
            seg = 6 + (int(hex_digest[2], 16) % 3)  # 6-8
            parts = [body[i:i+seg] for i in range(0, len(body), seg)]
            # Single or double underscore separator chosen deterministically
            sep = "__" if (int(hex_digest[3], 16) & 1) else "_"
            return "_" + sep.join(parts)

        # Strategy 3: Unicode homoglyphs (Greek/Cyrillic lookalikes) to confuse readers
        if self.scheme == 3:
            homoglyphs = [
                "o","ο","О","о","c","с","e","е","a","а","p","р",
                "x","х","y","у","l","I","І","і","k","κ"
            ]
            chars: list[str] = []
            for i, h in enumerate(hex_digest):
                idx = (int(h, 16) + i) % len(homoglyphs)
                chars.append(homoglyphs[idx])
            base_length = 24 + (int(hex_digest[4], 16) % 3) * 8
            length = int(base_length * self._get_length_multiplier())
            body = "".join(chars)[:length]
            seg = 7  # fixed segment for Unicode strategy
            parts = [body[i:i+seg] for i in range(0, len(body), seg)]
            return "_" + "_".join(parts)

        # Strategy 4: meaningful words from a dictionary (e.g., medical terms)
        if self.scheme == 4:
            words = self.dict_words
            if not words:
                # fallback just in case
                words = _DEFAULT_MEDICAL_WORDS
            # deterministically choose words based on hash and length
            idx1 = int(hex_digest[:8], 16) % len(words)
            candidate = words[idx1]
            
            # Add more words for longer lengths
            if self.name_length == "long":
                idx2 = int(hex_digest[8:16], 16) % len(words)
                candidate = f"{words[idx1]}_{words[idx2]}"
                # add a short suffix to drastically reduce collisions while deterministic
                suffix = hex_digest[16:19]
                candidate = candidate + "_" + suffix
            elif self.name_length == "medium":
                idx2 = int(hex_digest[8:16], 16) % len(words)
                candidate = f"{words[idx1]}_{words[idx2]}"
                # shorter suffix for medium length
                suffix = hex_digest[16:18]
                candidate = candidate + "_" + suffix
            # For short length, just use one word with a short suffix
            else:  # short
                suffix = hex_digest[16:18]
                candidate = candidate + "_" + suffix
            
            return candidate

        # Strategy 5: 2-3 meaningful words plus misleading role suffix
        if self.scheme == 5:
            words = self.dict_words
            n = len(words) if len(words) > 0 else len(_DEFAULT_MEDICAL_WORDS)
            def pick(off: int) -> str:
                base = int(hex_digest[off:off+8], 16)
                return words[base % n] if n else _DEFAULT_MEDICAL_WORDS[base % len(_DEFAULT_MEDICAL_WORDS)]
            w1 = pick(0)
            w2 = pick(8)
            
            # Adjust number of words and suffix based on length
            if self.name_length == "short":
                # Just one word with short suffix
                suffixes = ("mgr","fac","svc","adp","prv")
                suf = suffixes[int(hex_digest[24:26], 16) % len(suffixes)]
                return "_" + w1 + "_" + suf
            elif self.name_length == "medium":
                # Two words with medium suffix
                suffixes = ("mgr","factory","service","adapter","provider")
                suf = suffixes[int(hex_digest[24:26], 16) % len(suffixes)]
                return "_" + w1 + "_" + w2 + "_" + suf
            else:  # long
                # Original behavior: 2-3 words with full suffix
                use_third = (int(hex_digest[3], 16) & 1) == 1
                w3 = pick(16) if use_third else None
                suffixes = ("manager","factory","service","adapter","provider")
                suf = suffixes[int(hex_digest[24:26], 16) % len(suffixes)]
                parts = [w1, w2] + ([w3] if w3 else [])
                return "_" + "_".join(parts) + "_" + suf

        # Strategy 6: leetspeak with embedded digits (never at start)
        if self.scheme == 6:
            leet = {
                'a':'4','e':'3','i':'1','o':'0','s':'5','t':'7','b':'8','g':'9','z':'2','l':'1'
            }
            letters = "abcdefghijklmnopqrstuvwxyz"
            chars: list[str] = []
            for i in range(0, len(hex_digest), 2):
                val = int(hex_digest[i:i+2], 16)
                c = letters[val % 26]
                c = leet.get(c, c) if (val & 4) else c
                chars.append(c)
            base_length = 22 + (int(hex_digest[1], 16) % 10)  # 22..31
            length = int(base_length * self._get_length_multiplier())
            body = "".join(chars)[:length]
            # Inject extra digits at deterministic positions (avoid first char)
            inject_pos = 1 + (int(hex_digest[2], 16) % max(1, len(body)-1))
            body = body[:inject_pos] + str(int(hex_digest[3], 16) % 10) + body[inject_pos:]
            return "_" + body

        # Strategy 7: Greek letters sequence (valid identifier letters)
        if self.scheme == 7:
            greek = "αβγδεζηθικλμνξοπρστυφχψω"
            chars: list[str] = []
            for i, h in enumerate(hex_digest):
                idx = (int(h, 16) + i) % len(greek)
                chars.append(greek[idx])
            base_length = 20 + (int(hex_digest[0], 16) % 12)  # 20..31
            length = int(base_length * self._get_length_multiplier())
            body = "".join(chars)[:length]
            return "_" + body

        # Strategy 8: maximal confusables (Latin/Greek/Cyrillic lookalikes + digits for non-leading)
        if self.scheme == 8:
            # Start characters must be letters (no digits). Use lookalikes across scripts.
            start_pool = (
                # Latin
                'l','I','O','o','A','B','C','E','H','K','M','N','P','T','X','Y','Z',
                # Greek lookalikes
                'Α','Β','Ε','Η','Ι','Κ','Μ','Ν','Ο','Ρ','Τ','Χ','Υ','Ζ','ο','Ο',
                # Cyrillic lookalikes
                'А','В','Е','К','М','Н','О','Р','С','Т','У','Х','а','е','о','р','с','у','х'
            )
            continue_pool = start_pool + (
                # Add digits that look like letters for non-leading positions
                '0','1'
            )
            # Build body deterministically
            base_length = 28 + (int(hex_digest[5], 16) % 9)  # 28..36
            length = int(base_length * self._get_length_multiplier())
            first_idx = int(hex_digest[:2], 16) % len(start_pool)
            out_chars: list[str] = [start_pool[first_idx]]
            pos = 1
            while len(out_chars) < length:
                h = hex_digest[(pos * 2) % len(hex_digest):(pos * 2 + 2) % len(hex_digest) or None]
                idx = int(h, 16) % len(continue_pool)
                out_chars.append(continue_pool[idx])
                pos += 1
            return "_" + "".join(out_chars)

        # Strategy 9: alternating I/l pattern with varying lengths (IlIlIIlIll style)
        if self.scheme == 9:
            # Determine base length (8-20 characters)
            base_length = 8 + (int(hex_digest[0], 16) % 13)
            length = int(base_length * self._get_length_multiplier())
            
            # Create pattern of alternating I and l with some variation
            chars: list[str] = []
            current_char = 'I' if (int(hex_digest[1], 16) & 1) else 'l'
            
            for i in range(length):
                # Use hash bits to determine character and when to switch
                hash_byte = int(hex_digest[(i // 4) % len(hex_digest):(i // 4 + 1) % len(hex_digest) or None], 16)
                bit_pos = i % 4
                
                # Switch character based on hash bits
                if (hash_byte >> bit_pos) & 1:
                    current_char = 'l' if current_char == 'I' else 'I'
                
                chars.append(current_char)
                
                # Occasionally add extra characters for more variation
                if (hash_byte >> (bit_pos + 2)) & 1 and i < length - 1:
                    chars.append(current_char)
            
            # Add some random extra characters at the end for more confusion
            extra_chars = int(hex_digest[2], 16) % 4  # 0-3 extra characters
            for i in range(extra_chars):
                extra_char = 'I' if (int(hex_digest[3 + i], 16) & 1) else 'l'
                chars.append(extra_char)
            
            return "".join(chars)

        # Strategy 11: alternating I/l pattern with same length as original name
        if self.scheme == 11:
            # Use the original name length
            original_length = len(name)
            
            # Create pattern of alternating I and l with same length as original
            chars: list[str] = []
            current_char = 'I' if (int(hex_digest[1], 16) & 1) else 'l'
            
            for i in range(original_length):
                # Use hash bits to determine character and when to switch
                hash_byte = int(hex_digest[(i // 4) % len(hex_digest):(i // 4 + 1) % len(hex_digest) or None], 16)
                bit_pos = i % 4
                
                # Switch character based on hash bits
                if (hash_byte >> bit_pos) & 1:
                    current_char = 'l' if current_char == 'I' else 'I'
                
                chars.append(current_char)
            
            return "".join(chars)

        # Fallback to strategy 1 if unknown
        trans = str.maketrans({
            "a":"l","b":"I","c":"O","d":"o","e":"0","f":"1",
            "0":"O","1":"l","2":"I","3":"o","4":"0","5":"1",
            "6":"O","7":"l","8":"I","9":"o"
        })
        mapped = hex_digest.translate(trans)
        base_length = 24 + (int(hex_digest[0], 16) % 3) * 8
        length = int(base_length * self._get_length_multiplier())
        body = mapped[:length]
        segments = [body[i:i+8] for i in range(0, len(body), 8)]
        return "_" + "_".join(segments)

    def _should_rename_identifier(self, name: str) -> bool:
        if not isinstance(name, str) or not name or not name.isidentifier():
            return False
        if name in self.reserved or name in self.imported:
            return False
        # Never rename protected regex method names anywhere
        if name in getattr(self, '_protected_regex_methods', set()):
            return False
        if self._is_dunder(name):
            return False
        # Skip special method names and parameter names that are commonly used
        protected_names = {
            'cls', 'self', 'name', 'bases', 'dict', 'metaclass', 'module', 'qualname', 'doc',
            'args', 'kwargs', 'key', 'value', 'id', 'type', 'class', 'style', 'title', 'content', 'data',
            'reverse', 'start', 'maxsize', 'default', 'encoding', 'errors', 'newline', 'mode',
            # Standard library function parameters
            'days', 'seconds', 'microseconds', 'milliseconds', 'hours', 'minutes', 'weeks',
            'years', 'months', 'timezone', 'format', 'locale', 'base', 'radix', 'precision',
            'width', 'fill', 'align', 'sign', 'zero', 'grouping', 'decimal', 'thousands',
            'date', 'time', 'datetime', 'timestamp', 'epoch', 'utc', 'local', 'tzinfo'
        }
        if name in protected_names:
            return False
        return True

    def _rename(self, name: str, kind: Optional[str] = None) -> str:
        if not self._should_rename_identifier(name):
            return name
        if name not in self.map:
            new = self._gen_name(name, kind)
            # Avoid collisions with reserved/imported and already used new names
            if self.scheme == 11:
                # For scheme 11, maintain original length even with collisions
                original_length = len(name)
                collision_count = 0
                max_attempts = 50  # Prevent infinite loops
                
                while new in self.reserved or new in self.imported or new in self.map.values():
                    collision_count += 1
                    
                    # If we've tried too many times, fall back to standard collision handling
                    if collision_count > max_attempts:
                        new = "_" + new + "x"
                        break
                    
                    # Generate a new name with same length by modifying the seed
                    modified_seed = self.seed + f"_collision_{collision_count}"
                    seed_input = (modified_seed + ":" + name).encode()
                    hex_digest = hashlib.sha256(seed_input).hexdigest()
                    
                    # Recreate the I/l pattern with same length
                    chars: list[str] = []
                    current_char = 'I' if (int(hex_digest[1], 16) & 1) else 'l'
                    
                    for i in range(original_length):
                        hash_byte = int(hex_digest[(i // 4) % len(hex_digest):(i // 4 + 1) % len(hex_digest) or None], 16)
                        bit_pos = i % 4
                        
                        if (hash_byte >> bit_pos) & 1:
                            current_char = 'l' if current_char == 'I' else 'I'
                        
                        chars.append(current_char)
                    
                    new = "".join(chars)
                    
                    # If still colliding after many attempts, try length ±1
                    if collision_count > 20:
                        for length_offset in [1, -1]:
                            if original_length + length_offset > 0:
                                chars = []
                                current_char = 'I' if (int(hex_digest[1], 16) & 1) else 'l'
                                
                                for i in range(original_length + length_offset):
                                    hash_byte = int(hex_digest[(i // 4) % len(hex_digest):(i // 4 + 1) % len(hex_digest) or None], 16)
                                    bit_pos = i % 4
                                    
                                    if (hash_byte >> bit_pos) & 1:
                                        current_char = 'l' if current_char == 'I' else 'I'
                                    
                                    chars.append(current_char)
                                
                                new = "".join(chars)
                                if new not in self.reserved and new not in self.imported and new not in self.map.values():
                                    break
            else:
                # For other schemes, use the original collision handling
                while new in self.reserved or new in self.imported or new in self.map.values():
                    new = "_" + new + "x"
            self.map[name] = new
            # Track this rename for consistency
            self._variable_renames[name] = new
        return self.map[name]

    def visit_Import(self, node: ast.Import) -> ast.AST:
        for alias in node.names:
            self.imported.add(alias.asname or alias.name.split(".")[0])
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        for alias in node.names:
            self.imported.add(alias.asname or alias.name)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self._annotation_depth > 0: return node
        if isinstance(node.ctx, ast.Load) and node.id not in self.defined and node.id not in self.imported:
            # Check if this is a class name that was renamed
            if node.id in self.map:
                node.id = self.map[node.id]
            return node
        
        # Check if this name should be renamed
        if self._should_rename_identifier(node.id):
            rename_kind = "var" if isinstance(node.ctx, (ast.Store, ast.Del)) else None
            node.id = self._rename(node.id, kind=rename_kind)
        
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        node.value = self.visit(node.value)
        
        # Don't rename built-in attributes and methods
        builtin_attrs = {
            'now', 'today', 'utcnow', 'fromtimestamp', 'fromisoformat', 'strftime', 'strptime',
            'replace', 'astimezone', 'utcoffset', 'dst', 'tzname', 'isoformat', 'ctime',
            'date', 'time', 'timetz', 'utctimetuple', 'timetuple', 'timestamp',
            'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'tzinfo',
            'days', 'seconds', 'microseconds', 'total_seconds', 'max', 'min', 'resolution',
            # Complex number attributes
            'real', 'imag', 'conjugate',
            # String attributes and methods
            'startswith', 'endswith', 'strip', 'split', 'join', 'replace', 'find', 'index',
            'upper', 'lower', 'title', 'capitalize', 'isalpha', 'isdigit', 'isalnum',
            # List attributes and methods
            'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'copy', 'sort', 'reverse',
            'count', 'index', 'len',
            # Dictionary attributes and methods
            'keys', 'values', 'items', 'get', 'setdefault', 'update', 'popitem', 'clear',
            # Set attributes and methods
            'add', 'remove', 'discard', 'union', 'intersection', 'difference', 'symmetric_difference',
            # File attributes and methods
            'read', 'write', 'close', 'seek', 'tell', 'flush', 'truncate',
            # Database methods
            'cursor', 'execute', 'fetchone', 'fetchall', 'fetchmany', 'commit', 'rollback',
            'connect', 'close', 'rowcount', 'lastrowid', 'description',
            # Document and object attributes
            'text', 'content', 'body', 'title', 'author', 'subject', 'paragraphs', 'sections',
            'tables', 'pictures', 'shapes', 'headers', 'footers', 'page_break', 'line_break',
            'bold', 'italic', 'underline', 'font', 'size', 'color', 'alignment', 'indent',
            'spacing', 'margin', 'border', 'background', 'foreground', 'style', 'format',
            # Object attributes
            'name', 'value', 'id', 'type', 'class', 'doc', 'module', 'qualname',
            # Regular expression methods (e.g., re, compiled pattern objects)
            'match', 'findall', 'split', 'sub'
        }
        
        # Check if this attribute should be renamed
        # First, check if this is a built-in attribute that should never be renamed
        if node.attr in builtin_attrs:
            # Built-in attributes should never be renamed
            return node
        
        # Check if this is a module attribute (like csv.writer, json.dumps) that should not be renamed
        if isinstance(node.value, ast.Name) and node.value.id in self.imported:
            # This is an attribute of an imported module, don't rename it
            return node
        
        # Then check if this attribute should be renamed as a method or variable
        if node.attr in self.map:
            # This is a method name or variable that was defined in this module, so rename it
            node.attr = self.map[node.attr]
        elif node.attr in self._global_method_renames:
            # This is a method call to a renamed method, use the global rename
            node.attr = self._global_method_renames[node.attr]
        elif node.attr in self._method_renames:
            # This is a method call, use the class-specific rename if available
            node.attr = self._method_renames[node.attr]
        elif self._in_class_context and node.attr in self._variable_renames:
            # This is an instance attribute that should be renamed for consistency with local variables
            node.attr = self._variable_renames[node.attr]
        elif hasattr(node.value, 'id') and node.value.id == 'self':
            # This is a self.attribute access - check if the attribute name should be renamed globally
            if node.attr in self._global_attribute_renames:
                node.attr = self._global_attribute_renames[node.attr]
            elif node.attr in self.map:
                node.attr = self.map[node.attr]
            elif node.attr in self._variable_renames:
                node.attr = self._variable_renames[node.attr]
        
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node.names = [self._rename(n, kind="var") for n in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        node.names = [self._rename(n, kind="var") for n in node.names]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Rename function name if not in class context
        if not self._in_class_context:
            node.name = self._rename(node.name, kind="function")
        
        node.args = self.visit(node.args)
        if node.returns is not None:
            self._annotation_depth += 1
            try:
                node.returns = self.visit(node.returns)
            finally:
                self._annotation_depth -= 1
        node.body = [self.visit(n) for n in node.body]
        new_decorators = []
        for d in node.decorator_list:
            if isinstance(d, ast.Name):
                new_decorators.append(ast.Name(id=self._rename(d.id), ctx=d.ctx))
            else:
                new_decorators.append(self.visit(d))
        node.decorator_list = new_decorators
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        # Rename function name if not in class context (consistent with FunctionDef)
        if not self._in_class_context:
            node.name = self._rename(node.name, kind="function")
        node.args = self.visit(node.args)
        if node.returns is not None:
            self._annotation_depth += 1
            try:
                node.returns = self.visit(node.returns)
            finally:
                self._annotation_depth -= 1
        node.body = [self.visit(n) for n in node.body]
        new_decorators = []
        for d in node.decorator_list:
            if isinstance(d, ast.Name):
                new_decorators.append(ast.Name(id=self._rename(d.id), ctx=d.ctx))
            else:
                new_decorators.append(self.visit(d))
        node.decorator_list = new_decorators
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        # Rename the class name
        old_class_name = node.name
        node.name = self._rename(old_class_name, kind="class")
        self.class_names.add(old_class_name)  # Keep track of original name for reference
        self.class_names.add(node.name)  # Also track the new name
        
        # Set class context flag and clear current class methods before processing class body
        was_in_class = self._in_class_context
        was_current_class_methods = self._current_class_methods.copy()
        was_method_renames = self._method_renames.copy()
        was_variable_renames = self._variable_renames.copy()
        self._in_class_context = True
        self._current_class_methods.clear()
        self._method_renames.clear()
        self._variable_renames.clear()
        
        # Method renames are already collected in the first pass, just set up class context
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                old_name = item.name
                # If method was protected, it won't be in global map; skip renaming
                if old_name in self._protected_regex_methods:
                    continue
                new_name = self._global_method_renames.get(old_name, self._rename(old_name, kind="function"))
                self._current_class_methods.add(old_name)
                self._method_renames[old_name] = new_name
                # Actually rename the method name
                item.name = new_name
        
        # Second pass: collect all variable names and instance attributes that will be renamed
        # This ensures we have the complete mapping before processing method bodies
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Process the method body to collect variable renames
                for stmt in item.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                if self._should_rename_identifier(target.id):
                                    # This variable will be renamed, track it
                                    self._variable_renames[target.id] = self._rename(target.id, kind="var")
                            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                # This is self.attribute_name assignment, track the attribute name globally
                                if self._should_rename_identifier(target.attr):
                                    self._global_attribute_renames[target.attr] = self._rename(target.attr, kind="var")
        
        # Second pass: process the method bodies with the complete rename mapping
        try:
            node.body = [self.visit(n) for n in node.body]
        finally:
            # Restore previous context
            self._in_class_context = was_in_class
            self._current_class_methods = was_current_class_methods
            self._method_renames = was_method_renames
            self._variable_renames = was_variable_renames
            
        new_decorators = []
        for d in node.decorator_list:
            if isinstance(d, ast.Name):
                new_decorators.append(ast.Name(id=self._rename(d.id), ctx=d.ctx))
            else:
                new_decorators.append(self.visit(d))
        node.decorator_list = new_decorators
        return node

    def visit_arguments(self, node: ast.arguments) -> ast.AST:
        for field in ("posonlyargs", "args", "kwonlyargs"):
            setattr(node, field, [self.visit(a) for a in getattr(node, field, [])])
        if node.vararg:
            node.vararg = self.visit(node.vararg)
        if node.kwarg:
            node.kwarg = self.visit(node.kwarg)
        node.defaults = [self.visit(d) for d in node.defaults]
        node.kw_defaults = [self.visit(d) if d is not None else None for d in node.kw_defaults]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        old_name = node.arg
        # Don't rename special argument names and common parameter names
        protected_params = {
            'args', 'kwargs', 'cls', 'self', 'name', 'value', 'key', 'reverse',
            'id', 'type', 'class', 'style', 'title', 'content', 'data',
            'default', 'encoding', 'errors', 'newline', 'mode',
            # Standard library function parameters
            'days', 'seconds', 'microseconds', 'milliseconds', 'hours', 'minutes', 'weeks',
            'years', 'months', 'timezone', 'format', 'locale', 'base', 'radix', 'precision',
            'width', 'fill', 'align', 'sign', 'zero', 'grouping', 'decimal', 'thousands',
            'date', 'time', 'datetime', 'timestamp', 'epoch', 'utc', 'local', 'tzinfo'
        }
        if old_name not in protected_params:
            node.arg = self._rename(old_name, kind="var")
        if node.annotation:
            self._annotation_depth += 1
            try:
                node.annotation = self.visit(node.annotation)
            finally:
                self._annotation_depth -= 1
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        if node.arg is not None:
            current = self._call_func_stack[-1] if self._call_func_stack else None
            
            # Check if this keyword argument was renamed as a parameter
            if node.arg in self.map:
                # This keyword argument corresponds to a renamed parameter, use the same rename
                node.arg = self.map[node.arg]
            else:
                # Don't rename keyword arguments for built-in functions and special methods
                should_rename = True
                
                # Built-in function keywords that should not be renamed
                if current in {'min', 'max', 'sorted', 'sort'} and node.arg in {'key', 'reverse'}:
                    should_rename = False
                elif current == 'enumerate' and node.arg == 'start':
                    should_rename = False
                elif current == 'lru_cache' and node.arg == 'maxsize':
                    should_rename = False
                # Special method keywords that should not be renamed
                elif node.arg in {'cls', 'self', 'name', 'bases', 'dict', 'metaclass', 'module', 'qualname', 'doc'}:
                    should_rename = False
                # Don't rename keyword arguments for object creation methods
                elif current in {'__new__', '__init__', '__call__'}:
                    should_rename = False
                # Don't rename common keyword argument names that are often used as parameters
                elif node.arg in {'key', 'value', 'name', 'id', 'type', 'class', 'style', 'title', 'content', 'data',
                                 'days', 'seconds', 'microseconds', 'milliseconds', 'hours', 'minutes', 'weeks',
                                 'years', 'months', 'timezone', 'format', 'locale', 'base', 'radix', 'precision',
                                 'width', 'fill', 'align', 'sign', 'zero', 'grouping', 'decimal', 'thousands',
                                 'date', 'time', 'datetime', 'timestamp', 'epoch', 'utc', 'local', 'tzinfo',
                                 # NumPy and statistical function parameters
                                 'rowvar', 'bias', 'ddof', 'keepdims', 'axis', 'out', 'dtype', 'order',
                                 'subok', 'copy', 'ndmin', 'casting', 'where', 'initial', 'before', 'after',
                                 'mode', 'size', 'fill_value', 'method', 'tolerance', 'interpolation',
                                 'side', 'kind', 'sorter', 'na_option', 'ascending', 'inplace', 'ignore_index',
                                 'level', 'sort_remaining', 'na_position', 'ignore_na', 'pct', 'numeric_only'}:
                    should_rename = False
                # Common filesystem/API keywords that must remain stable
                elif node.arg in {'exist_ok'}:
                    should_rename = False
                
                if should_rename:
                    node.arg = self._rename(node.arg, kind="var")
                
        if node.value:
            node.value = self.visit(node.value)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        def _fname(expr: ast.AST) -> Optional[str]:
            if isinstance(expr, ast.Name):
                return expr.id
            if isinstance(expr, ast.Attribute):
                return expr.attr
            return None
        
        self._call_func_stack.append(_fname(node.func))
        
        # Process the function expression
        node.func = self.visit(node.func)
        
        # If the function is an Attribute and the attribute name should be renamed, rename it
        if isinstance(node.func, ast.Attribute):
            # Do not rename attributes on imported modules (e.g., re.findall)
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.imported:
                pass
            else:
                # Protect common regular-expression methods (module and compiled pattern)
                protected_regex_methods = {
                    'match', 'search', 'fullmatch', 'findall', 'finditer', 'split', 'sub', 'subn', 'compile'
                }
                if node.func.attr in protected_regex_methods:
                    pass
                elif node.func.attr in self._global_method_renames:
                    node.func.attr = self._global_method_renames[node.func.attr]
        
        node.args = [self.visit(a) for a in node.args]
        node.keywords = [self.visit(k) for k in node.keywords]
        self._call_func_stack.pop()
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        node.args = self.visit(node.args)
        node.body = self.visit(node.body)
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        # Rename exception alias: `except E as name:`
        if node.name:
            node.name = self._rename(node.name, kind="var")
        node.type = self.visit(node.type) if node.type else None
        node.body = [self.visit(n) for n in node.body]
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        # x: Type = value
        node.target = self.visit(node.target)
        self._annotation_depth += 1
        try:
            node.annotation = self.visit(node.annotation)
        finally:
            self._annotation_depth -= 1
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

def obfuscate_code(source: str, seed: str = "obf", scheme: int = 1, name_length: str = "long", dictionary_words: Optional[Sequence[str]] = None) -> Tuple[str, List[Tuple[str, str]]]:
    """Obfuscate variable and function names; preserve imports like gcd/cache/Counter.

    Automatically respects imports declared in a global prelude if available
    (e.g., `utils.util_crux_eval.BASE_IMPORTS`).

    Parameters
    - source: Python source code to obfuscate
    - seed: deterministic obfuscation seed.
    - scheme: select identifier style:
        1 = ASCII with l/I/o/O/0/1 mapping (default),
        2 = highly ambiguous ASCII (l/I/1/O/0/o with jittered underscores),
        3 = Unicode homoglyphs (Greek/Cyrillic lookalikes),
        4 = meaningful words from a provided dictionary (defaults to medical terms),
        5 = 2-3 meaningful words plus misleading role suffix,
        6 = leetspeak with embedded digits,
        7 = Greek letters sequence,
        8 = maximal Unicode confusables (hardest to visually differentiate),
        9 = alternating I/l pattern with varying lengths (IlIlIIlIll style),
        10 = single-character identifiers (from a large Unicode letter pool),
        11 = alternating I/l pattern with same length as original name,
        12 = type-based counters (classN/functionN/varN),
        13 = misleading names by type (from predefined pools).
    - name_length: control the length of obfuscated names:
        "short" = approximately 40% of original length,
        "medium" = approximately 70% of original length,
        "long" = original length (default).
    - dictionary_words: optional custom word list for scheme 4.
    
    Returns
    - Tuple containing:
        - obfuscated_code: the obfuscated Python source code
        - rename_mappings: list of (old_name, new_name) tuples showing what was renamed
    """
    tree = ast.parse(source)
    defined = _collect_defined_names(tree)
    pre_imported: Set[str] = set()
    try:
        from .util_crux_eval import BASE_IMPORTS as _BASE
        names, modules = _collect_imports(_BASE)
        pre_imported |= names | modules
    except Exception:
        pass
    renamer = _Renamer(seed=seed, scheme=scheme, name_length=name_length, dictionary_words=dictionary_words, defined_names=defined, pre_imported=pre_imported)
    
    # First pass: collect all method renames from all classes
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            renamer._collect_class_method_renames(node)
    
    # Second pass: process the entire AST with method renames available
    new_tree = renamer.visit(tree)
    ast.fix_missing_locations(new_tree)
    
    # Create list of rename mappings
    rename_mappings = list(renamer.map.items())
    
    try:
        # Python 3.9+: ast.unparse
        obfuscated_code = ast.unparse(new_tree)
        return obfuscated_code, rename_mappings
    except AttributeError:
        # For very old Pythons without ast.unparse, fall back to astor if available.
        try:
            import astor  # type: ignore
            obfuscated_code = astor.to_source(new_tree)
            return obfuscated_code, rename_mappings
        except Exception as e:
            raise RuntimeError(
                "ast.unparse not available and astor not installed."
            ) from e

def _collect_imports(source: str) -> Tuple[Set[str], Set[str]]:
    """Collect imported simple names from a source prelude to pre-reserve them.

    Returns (imported_names, imported_modules).
    """
    try:
        tree = ast.parse(source)
    except Exception:
        return set(), set()
    imported_names: Set[str] = set()
    imported_modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.asname or alias.name.split(".")[0])
                imported_modules.add(root)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
            if node.module:
                imported_modules.add(node.module.split(".")[0])
    return imported_names, imported_modules

