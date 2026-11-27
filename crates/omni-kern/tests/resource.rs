use omni_kern::resource::Budget;

#[test]
fn budget_converts_gib_to_tokens_pages_and_table() {
    let budget = Budget::new(1.0, 1, 16, 8);
    assert_eq!(budget.tokens, 1 << 30);
    assert_eq!(budget.pages, (1 << 30) / 16 + 1);
    assert_eq!(budget.page_table, budget.pages + 8);
    assert_eq!(budget.max_seqs, 8);
}

use kern_pool::Denied;
use omni_kern::resource::admit;

#[test]
fn admit_classifies_busy_and_exceeds_pool() {
    let budget = Budget { tokens: 0, pages: 10, page_table: 0, max_seqs: 0 };
    assert_eq!(admit(10, 1, 1, &budget), Ok(()));
    assert_eq!(admit(11, 1, 1, &budget), Err(Denied::Busy));
    assert_eq!(admit(11, 9, 1, &budget), Err(Denied::Busy));
    assert_eq!(admit(11, 10, 1, &budget), Err(Denied::ExceedsPool));
}
