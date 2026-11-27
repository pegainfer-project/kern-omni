use omni_kern::resource::Budget;

#[test]
fn budget_converts_gib_to_tokens_pages_and_table() {
    let budget = Budget::new(1.0, 1, 16, 8);
    assert_eq!(budget.tokens, 1 << 30);
    assert_eq!(budget.pages, (1 << 30) / 16 + 1);
    assert_eq!(budget.page_table, budget.pages + 8);
    assert_eq!(budget.max_seqs, 8);
}
