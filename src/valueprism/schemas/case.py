# Copied from values-schemas/values_schemas/case.py.
from pydantic import BaseModel, model_validator
from typing import Literal

ValueAlignmentStatus = Literal['promotes', 'violates', 'neutral']

# List of all value names used in ethical dilemmas
VALUE_NAMES = ["autonomy", "beneficence", "nonmaleficence", "justice"]


# =============================================================================
# Helper functions for value tag validation
# =============================================================================

def get_value_tags(choice_1: "ChoiceWithValues", choice_2: "ChoiceWithValues") -> dict[str, tuple[str, str]]:
    """Extract value tags from both choices into a dictionary.
    
    Args:
        choice_1: First choice with value alignment tags
        choice_2: Second choice with value alignment tags
        
    Returns:
        Dictionary mapping value name to (choice_1_tag, choice_2_tag) tuple.
        Example: {"autonomy": ("promotes", "violates"), "beneficence": ("neutral", "promotes")}
    """
    return {
        value: (getattr(choice_1, value), getattr(choice_2, value))
        for value in VALUE_NAMES
    }


def is_valid_per_value_pattern(tag_1: str, tag_2: str) -> bool:
    """Check if a single value's tag combination is valid.
    
    Valid patterns:
    - (neutral, neutral): Value not engaged
    - (neutral, promotes/violates): One-sided engagement
    - (promotes/violates, neutral): One-sided engagement  
    - (promotes, violates): Classic opposition
    - (violates, promotes): Classic opposition
    
    Invalid patterns:
    - (promotes, promotes): Same direction - no tension
    - (violates, violates): Same direction - no tension
    
    Args:
        tag_1: Tag for choice 1 (promotes/violates/neutral)
        tag_2: Tag for choice 2 (promotes/violates/neutral)
        
    Returns:
        True if the pattern is valid, False otherwise.
    """
    # Invalid if both choices have the same non-neutral tag
    if tag_1 == tag_2 and tag_1 != "neutral":
        return False
    return True


def count_choice_effects(tags: dict[str, tuple[str, str]]) -> tuple[bool, bool, bool, bool]:
    """Count whether each choice has promotions and/or violations.
    
    Args:
        tags: Dictionary mapping value name to (choice_1_tag, choice_2_tag)
        
    Returns:
        Tuple of (c1_promotes, c1_violates, c2_promotes, c2_violates) booleans
        indicating whether each choice has at least one promotion/violation.
    """
    c1_promotes = any(t1 == "promotes" for t1, t2 in tags.values())
    c1_violates = any(t1 == "violates" for t1, t2 in tags.values())
    c2_promotes = any(t2 == "promotes" for t1, t2 in tags.values())
    c2_violates = any(t2 == "violates" for t1, t2 in tags.values())
    
    return (c1_promotes, c1_violates, c2_promotes, c2_violates)


def has_cross_value_opposition(tags: dict[str, tuple[str, str]]) -> bool:
    """Check if there is cross-value opposition between choices.
    
    At least ONE of these must be true for valid opposition:
    1. Different values promoted: Choice 1 promotes value X AND Choice 2 promotes value Y (X ≠ Y)
    2. Different values violated: Choice 1 violates value X AND Choice 2 violates value Y (X ≠ Y)
    3. Same value in opposition: For some value, one choice promotes and other violates
    
    Args:
        tags: Dictionary mapping value name to (choice_1_tag, choice_2_tag)
        
    Returns:
        True if there is valid cross-value opposition, False otherwise.
    """
    c1_promotes = [v for v, (t1, t2) in tags.items() if t1 == "promotes"]
    c2_promotes = [v for v, (t1, t2) in tags.items() if t2 == "promotes"]
    c1_violates = [v for v, (t1, t2) in tags.items() if t1 == "violates"]
    c2_violates = [v for v, (t1, t2) in tags.items() if t2 == "violates"]
    
    # Condition 1: Different values promoted by each choice
    if c1_promotes and c2_promotes and set(c1_promotes) != set(c2_promotes):
        return True
    
    # Condition 2: Different values violated by each choice
    if c1_violates and c2_violates and set(c1_violates) != set(c2_violates):
        return True
    
    # Condition 3: Same value in opposition (promotes vs violates)
    for v, (t1, t2) in tags.items():
        if (t1 == "promotes" and t2 == "violates") or (t1 == "violates" and t2 == "promotes"):
            return True
    
    return False


def has_balance_violation(tags: dict[str, tuple[str, str]]) -> bool:
    """Check if there is a 'free lunch' / balance violation.
    
    A balance violation occurs when one choice is clearly better than the other:
    - Pure upside vs pure downside: INVALID
    - Pure upside vs mixed: VALID  
    - Mixed vs mixed: VALID
    - Pure downside vs pure downside: VALID (lesser evil)
    - Mixed vs pure downside: INVALID (asymmetric - one has no upside)
    
    The rule is: If one choice has ANY violations and NO promotions (pure downside),
    the other choice must also have at least one violation OR no promotions.
    
    Args:
        tags: Dictionary mapping value name to (choice_1_tag, choice_2_tag)
        
    Returns:
        True if there IS a balance violation (invalid case), False if balanced (valid).
    """
    c1_promotes, c1_violates, c2_promotes, c2_violates = count_choice_effects(tags)
    
    # Classify each choice
    c1_pure_upside = c1_promotes and not c1_violates
    c1_pure_downside = c1_violates and not c1_promotes
    c2_pure_upside = c2_promotes and not c2_violates
    c2_pure_downside = c2_violates and not c2_promotes
    
    # Invalid: pure upside vs pure downside
    if (c1_pure_upside and c2_pure_downside) or (c1_pure_downside and c2_pure_upside):
        return True
    
    # Invalid: mixed vs pure downside (asymmetric - one has no upside)
    c1_mixed = c1_promotes and c1_violates
    c2_mixed = c2_promotes and c2_violates
    
    if (c1_mixed and c2_pure_downside) or (c2_mixed and c1_pure_downside):
        return True
    
    return False


class ChoiceWithValues(BaseModel):
    choice: str
    autonomy: ValueAlignmentStatus
    beneficence: ValueAlignmentStatus
    nonmaleficence: ValueAlignmentStatus
    justice: ValueAlignmentStatus


class BenchmarkCandidate(BaseModel):
    vignette: str
    choice_1: ChoiceWithValues
    choice_2: ChoiceWithValues
    
    @model_validator(mode='after')
    def validate_value_conflict(self) -> 'BenchmarkCandidate':
        """Validates that the case creates a genuine ethical dilemma.
        
        Four constraints must be satisfied:
        
        1. Per-value validity: Each value must have a valid pattern - no same-direction 
           tags (promotes+promotes or violates+violates are invalid).
           
        2. Minimum engagement: At least 2 values must be non-neutral.
        
        3. Cross-value opposition: There must be genuine tension between choices via:
           - Different values promoted by each choice, OR
           - Different values violated by each choice, OR  
           - Same value in opposition (one promotes, other violates)
           
        4. No free lunch: One choice cannot be "all upside" while the other is "all 
           downside". Both pure-upside vs pure-downside AND mixed vs pure-downside
           are invalid (the latter is asymmetric - one choice has no upside at all).
        
        Valid patterns include:
        - Classic cross-conflict: C1 promotes A & violates B, C2 violates A & promotes B
        - Cross-value tradeoffs: C1 promotes A, C2 promotes B (different values)
        - Lesser evil: C1 violates A, C2 violates B (both only violate different values)
        - One-sided engagement: A value can be promotes+neutral or violates+neutral
        """
        # Extract all value tags into a dictionary
        tags = get_value_tags(self.choice_1, self.choice_2)
        
        # =========================================================================
        # Constraint 1: Per-value validity (reject same-direction patterns)
        # =========================================================================
        for value, (tag_1, tag_2) in tags.items():
            if not is_valid_per_value_pattern(tag_1, tag_2):
                raise ValueError(
                    f"INVALID: Both choices '{tag_1}' {value}. "
                    f"FIX: For {value}, either (1) flip one choice to the opposite tag, or "
                    f"(2) set one choice to 'neutral'. "
                    f"Same-direction tags create no tension."
                )
        
        # =========================================================================
        # Constraint 2: Minimum 2 non-neutral values
        # =========================================================================
        involved_values = [
            value for value, (tag_1, tag_2) in tags.items()
            if tag_1 != "neutral" or tag_2 != "neutral"
        ]
        
        if len(involved_values) < 2:
            raise ValueError(
                f"INVALID: Only {len(involved_values)} value(s) engaged: {involved_values or 'none'}. "
                f"FIX: Tag at least 2 different values as non-neutral across the choices. "
                f"Example: Choice 1 promotes autonomy, Choice 2 promotes beneficence."
            )
        
        # =========================================================================
        # Constraint 3: Cross-value opposition
        # =========================================================================
        if not has_cross_value_opposition(tags):
            raise ValueError(
                "INVALID: No cross-value tension between choices. "
                "FIX: Ensure at least one of: "
                "(1) Each choice promotes a DIFFERENT value, "
                "(2) Each choice violates a DIFFERENT value, or "
                "(3) One value has promotes↔violates opposition. "
                "The choices must create a genuine tradeoff."
            )
        
        # =========================================================================
        # Constraint 4: No free lunch (balance check)
        # =========================================================================
        if has_balance_violation(tags):
            c1_promotes, c1_violates, c2_promotes, c2_violates = count_choice_effects(tags)
            
            # Identify which choice is problematic
            c1_pure_down = c1_violates and not c1_promotes
            c2_pure_down = c2_violates and not c2_promotes
            
            if c1_pure_down:
                raise ValueError(
                    "INVALID: Choice 1 only has violations (no promotions), making Choice 2 obviously better. "
                    "FIX: Give Choice 1 at least one 'promotes' tag, OR give Choice 2 only violations too (lesser-evil scenario)."
                )
            else:
                raise ValueError(
                    "INVALID: Choice 2 only has violations (no promotions), making Choice 1 obviously better. "
                    "FIX: Give Choice 2 at least one 'promotes' tag, OR give Choice 1 only violations too (lesser-evil scenario)."
                )
        
        return self
