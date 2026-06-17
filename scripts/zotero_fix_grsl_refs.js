/*
Batch-fix selected Zotero items for IEEE GRSL reference output.

Usage:
1. In Zotero, select the items you want to fix.
2. Open Tools -> Developer -> Run JavaScript.
3. Paste this file and run it.

What it does:
- Normalizes arXiv/preprint metadata so CSL can render "arXiv:xxxx.xxxxx"
- Marks likely early-access journal articles with "Status: early access"
- Removes fake "pp. 1-1" page ranges from early-access items
- Normalizes article-number-like page fields to plain digits
*/

(async () => {
  const zoteroPane = Zotero.getActiveZoteroPane();
  const items = (zoteroPane?.getSelectedItems() || []).filter(item => item.isRegularItem());

  if (!items.length) {
    return "No regular Zotero items selected.";
  }

  const normalizeDash = value => (value || "").replace(/\s*[-\u2013\u2014]\s*/g, "\u2013").trim();
  const clean = value => (value || "").replace(/\s+/g, " ").trim();
  const titleOf = item => clean(item.getField("title")) || `item-${item.id}`;

  const splitExtra = extra => {
    if (!extra) return [];
    return extra.split(/\r?\n/).filter(line => line.trim() !== "");
  };

  const escapeRegExp = value => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const updateExtraFields = (extra, updates) => {
    const keys = Object.keys(updates);
    const regexes = keys.map(key => ({
      key,
      re: new RegExp(`^${escapeRegExp(key)}\\s*:`, "i"),
    }));

    const kept = splitExtra(extra).filter(line => !regexes.some(({ re }) => re.test(line)));
    const prepended = [];
    for (const key of keys) {
      const value = updates[key];
      if (value !== null && value !== undefined && `${value}`.trim() !== "") {
        prepended.push(`${key}: ${value}`);
      }
    }
    return [...prepended, ...kept].join("\n");
  };

  const tryGetField = (item, field) => {
    try {
      return item.getField(field);
    }
    catch (_err) {
      return "";
    }
  };

  const trySetField = (item, field, value) => {
    try {
      item.setField(field, value);
      return true;
    }
    catch (_err) {
      return false;
    }
  };

  const arxivIDPatterns = [
    /10\.48550\/arXiv\.([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)$/i,
    /arXiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)/i,
    /\b([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b/i,
  ];

  const extractArxivID = item => {
    const candidates = [
      tryGetField(item, "DOI"),
      tryGetField(item, "number"),
      tryGetField(item, "publisher"),
      tryGetField(item, "url"),
      tryGetField(item, "extra"),
    ];

    for (const candidate of candidates) {
      const text = clean(candidate);
      if (!text) continue;
      for (const pattern of arxivIDPatterns) {
        const match = text.match(pattern);
        if (match) {
          return match[1];
        }
      }
    }
    return "";
  };

  const looksLikeArxiv = item => {
    const fields = [
      tryGetField(item, "DOI"),
      tryGetField(item, "number"),
      tryGetField(item, "publisher"),
      tryGetField(item, "url"),
      tryGetField(item, "extra"),
    ].map(clean).join(" ");
    return /arxiv/i.test(fields) || /10\.48550\/arXiv\./i.test(fields);
  };

  const articleNumberMatch = pages => {
    const value = clean(pages);
    if (!value) return "";
    const match = value.match(/^(?:art\.?\s*no\.?\s*|pp?\.?\s*)?([0-9]{4,})\.?$/i);
    return match ? match[1] : "";
  };

  const pageRangeMatch = pages => {
    const value = clean(pages);
    if (!value) return "";
    const match = value.match(/^(?:pp?\.?\s*)?([0-9A-Za-z]+(?:\s*[-\u2013\u2014]\s*[0-9A-Za-z]+)+)\.?$/i);
    return match ? normalizeDash(match[1]) : "";
  };

  const isLikelyEarlyAccess = item => {
    const itemType = Zotero.ItemTypes.getName(item.itemTypeID);
    if (itemType !== "journalArticle") return false;

    const pages = normalizeDash(tryGetField(item, "pages"));
    const volume = clean(tryGetField(item, "volume"));
    const issue = clean(tryGetField(item, "issue"));
    const extra = clean(tryGetField(item, "extra"));

    if (/^status:\s*early access$/im.test(extra)) return true;
    return /^1\u20131$/.test(pages) && !volume && !issue;
  };

  let changedCount = 0;
  const report = [];

  for (const item of items) {
    const itemNotes = [];
    let dirty = false;

    const itemType = Zotero.ItemTypes.getName(item.itemTypeID);
    const title = titleOf(item);

    let extra = tryGetField(item, "extra") || "";
    let pages = normalizeDash(tryGetField(item, "pages"));
    const volume = clean(tryGetField(item, "volume"));
    const issue = clean(tryGetField(item, "issue"));

    const singleArticleNumber = articleNumberMatch(pages);
    if (singleArticleNumber && pages !== singleArticleNumber) {
      trySetField(item, "pages", singleArticleNumber);
      pages = singleArticleNumber;
      dirty = true;
      itemNotes.push(`pages -> ${singleArticleNumber}`);
    }
    else {
      const range = pageRangeMatch(pages);
      if (range && pages !== range) {
        trySetField(item, "pages", range);
        pages = range;
        dirty = true;
        itemNotes.push(`pages normalized -> ${range}`);
      }
    }

    if (looksLikeArxiv(item)) {
      const arxivID = extractArxivID(item);
      const updates = {
        "Archive": "arXiv",
        "Archive Location": arxivID || null,
      };
      const nextExtra = updateExtraFields(extra, updates);
      if (nextExtra !== extra) {
        trySetField(item, "extra", nextExtra);
        extra = nextExtra;
        dirty = true;
        itemNotes.push(arxivID ? `arXiv normalized -> ${arxivID}` : "arXiv normalized");
      }

      const publisher = clean(tryGetField(item, "publisher"));
      if (/^arxiv$/i.test(publisher)) {
        trySetField(item, "publisher", "");
        dirty = true;
        itemNotes.push("publisher cleared");
      }

      const number = clean(tryGetField(item, "number"));
      if (/arxiv/i.test(number)) {
        trySetField(item, "number", arxivID || "");
        dirty = true;
        itemNotes.push(arxivID ? "number cleaned" : "number cleared");
      }
    }

    if (isLikelyEarlyAccess(item)) {
      const nextExtra = updateExtraFields(extra, {
        "Status": "early access",
      });
      if (nextExtra !== extra) {
        trySetField(item, "extra", nextExtra);
        extra = nextExtra;
        dirty = true;
        itemNotes.push("status -> early access");
      }

      if (/^1\u20131$/.test(pages) && itemType === "journalArticle" && !volume && !issue) {
        trySetField(item, "pages", "");
        pages = "";
        dirty = true;
        itemNotes.push("removed fake 1-1 pages");
      }
    }

    if (dirty) {
      await item.saveTx();
      changedCount += 1;
      report.push(`[changed] ${title}: ${itemNotes.join("; ")}`);
    }
    else {
      report.push(`[skip] ${title}`);
    }
  }

  return `Updated ${changedCount}/${items.length} item(s).\n\n${report.join("\n")}`;
})();
