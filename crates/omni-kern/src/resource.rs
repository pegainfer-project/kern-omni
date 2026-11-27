//! Paged-state resource helpers shared by the speech engines: the token
//! budget their KV pages come from, the pad lease every graph call names,
//! and the page/slot rows a paged call writes.

use std::collections::BTreeMap;
use std::ops::Range;

use anyhow::Context;
use anyhow::Result;
use kern_pool::Denied;
use kern_pool::Lease;
use kern_runtime::Runtime;

use crate::ints;

/// The paged-state budget a model's KV pages come from, and the bounds a
/// call's page table must fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub tokens: usize,
    /// The most pages the running sequences may hold.
    pub pages: usize,
    /// The longest page table one call writes: every running sequence's pages
    /// plus one page per padded row.
    pub page_table: usize,
    pub max_seqs: usize,
}

impl Budget {
    pub fn new(kv_gib: f64, bytes_per_token: usize, page: usize, max_seqs: usize) -> Self {
        let tokens = (kv_gib * (1u64 << 30) as f64) as usize / bytes_per_token;
        let pages = tokens.div_ceil(page) + 1;
        Self { tokens, pages, page_table: pages + max_seqs, max_seqs }
    }
}

/// The one lease a graph call pads rows with.
pub struct Pad(Lease);

impl Pad {
    pub fn new(rt: &mut Runtime) -> Result<Self> {
        Ok(Self(rt.lease(1)?))
    }

    pub fn lease(&self) -> &Lease {
        &self.0
    }

    pub fn pages(&self) -> &[i32] {
        self.0.page_ids()
    }
}

/// Lease `tokens` of paged state, refusing the same way every engine does
/// when the running sequences have taken the budget.
pub fn open(rt: &mut Runtime, tokens: usize, budget: &Budget, pad: &Pad) -> Result<std::result::Result<Lease, Denied>> {
    let lease = match rt.lease(tokens) {
        Ok(l) => l,
        Err(kern_runtime::Error::Denied(d)) => return Ok(Err(d)),
        Err(e) => return Err(e.into()),
    };
    if rt.pages_used() > budget.pages {
        let alone = lease.page_ids().len() + pad.pages().len();
        return Ok(Err(if alone > budget.pages { Denied::ExceedsPool } else { Denied::Busy }));
    }
    Ok(Ok(lease))
}

/// The rows of a paged call: one token position each, with the page table
/// its sequence's K/V lives in.
#[derive(Default)]
pub struct Rows {
    pub pos: Vec<i32>,
    pub slot: Vec<i32>,
    pub seq: Vec<i32>,
    pub indptr: Vec<i32>,
    pub pages: Vec<i32>,
}

impl Rows {
    pub fn push_seq(&mut self, lease: &Lease, positions: Range<usize>, s: usize) {
        if self.indptr.is_empty() {
            self.indptr.push(0);
        }
        for p in positions {
            self.pos.push(p as i32);
            self.slot.push(lease.slot(p) as i32);
            self.seq.push(s as i32);
        }
        self.pages.extend_from_slice(lease.page_ids());
        self.indptr.push(self.pages.len() as i32);
    }

    pub fn write(&self, rt: &mut Runtime, v: &BTreeMap<String, u64>) -> Result<()> {
        rt.write_input_at("t_pos", &ints(&self.pos), v)?;
        rt.write_input_at("t_slot", &ints(&self.slot), v)?;
        rt.write_input_at("t_seq", &ints(&self.seq), v)?;
        rt.write_input_at("kv_indptr", &ints(&self.indptr), v)?;
        rt.write_input_at("kv_pages", &ints(&self.pages), v).context("kv_pages")?;
        Ok(())
    }
}
