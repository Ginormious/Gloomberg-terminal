import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import time
import ssl
import json
from urllib.parse import urlparse, quote_plus

# Try newspaper4k
NEWSPAPER_AVAILABLE = False
try:
    from newspaper import Article as NewspaperArticle

    NEWSPAPER_AVAILABLE = True
except ImportError:
    pass

if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

# Main analyzer class
class GoogleNewsSentimentAnalyzer:
    def __init__(self):
        model_id = "ProsusAI/finbert"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        try:
            # Prefer safetensors so that python doesn't return an error message. Alternative is to just ignore the warning message entirely but this way works fine anyway.
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                trust_remote_code=False,
                use_safetensors=True
            )
        except Exception:
            # Last resort fallback to local models only
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                local_files_only=True,
                use_safetensors=True
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }

        # Initialize session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    # Resolve URL using Google's internal API
    def get_final_url(self, url: str, timeout: int = 10) -> Optional[str]:
        print(f"\nDEBUG: Resolving URL")
        print(f"Input URL: {url}")

        if not url:
            print("No URL provided to resolve")
            return None

        # Use Google's internal batchexecute API
        try:
            print(f"Using Google's batchexecute API")

            # First, get the page to extract the data-p attribute
            resp = requests.get(url, timeout=timeout)
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find the c-wiz element with data-p attribute
            c_wiz = soup.select_one('c-wiz[data-p]')
            if not c_wiz:
                print(f"Could not find c-wiz element with data-p")
                return None

            data_p = c_wiz.get('data-p')
            if not data_p:
                print(f"data-p attribute is empty")
                return None

            print(f"Found data-p attribute")

            # Parse the data-p JSON
            obj = json.loads(data_p.replace('%.@.', '["garturlreq",'))

            # Prepare the payload for Google's API
            payload = {
                'f.req': json.dumps([[['Fbv4je', json.dumps(obj[:-6] + obj[-2:]), 'null', 'generic']]])
            }

            headers = {
                'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            }

            api_url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
            print(f"Calling Google's batchexecute API")

            response = requests.post(api_url, headers=headers, data=payload, timeout=timeout)

            # Parse the response
            array_string = json.loads(response.text.replace(")]}'", ""))[0][2]
            article_url = json.loads(array_string)[1]

            print(f"Successfully resolved URL: {article_url}")
            return article_url

        except Exception as e:
            print(f"API method failed: {type(e).__name__}: {str(e)[:150]}")

        print(f"URL resolution failed")
        return None

    # Fetch Google News RSS results
    def fetch_google_news(self, query: str, max_results: int = 20) -> List[Dict]:
        print(f"Searching Google News for: {query}")

        feed_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(feed_url)

        if not feed.entries:
            print("No articles found.")
            return []

        results = []
        for entry in feed.entries[:max_results]:

            title = getattr(entry, "title", "")
            raw_link = getattr(entry, "link", "") or getattr(entry, "id", "")

            print(f"\nProcessing: {title}")
            print(f"Raw link: {raw_link}")

            # Resolve redirect
            final_url = self.get_final_url(raw_link)

            if final_url and "news.google.com" in final_url:
                print(f"URL still points to Google News, marking as unresolved")
                final_url = None
            # If unresolved, fallback to title+summary
            if not final_url:
                print(f"No valid URL - use title+summary fallback")
            else:
                print(f"Valid URL obtained: {final_url}")

            # Extract summary
            summary_text = ""
            if hasattr(entry, "summary"):
                summary_text = BeautifulSoup(entry.summary, "html.parser").get_text(" ", strip=True)

            # Source
            source = None
            if hasattr(entry, "source") and isinstance(entry.source, dict):
                source = entry.source.get("title")

            if not source:
                if final_url:
                    try:
                        source = urlparse(final_url).hostname
                    except Exception:
                        source = None
            if not source:
                source = "unknown"

            # Published
            published = getattr(entry, "published", "")
            if not published and hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6]).isoformat()

            results.append({
                "title": title,
                "url": final_url or "",
                "summary": summary_text,
                "source": source,
                "published": published,
                "feed_entry": entry
            })

        print(f"Retrieved {len(results)} items.\n")
        return results

    # Fetch across multiple queries (name, ticker, name+stock)
    def lookup_ticker_top1(self, company_name: str) -> Optional[str]:
        #Lookup ticker symbol via Yahoo Finance search API
        try:
            q = quote_plus(company_name)
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={q}"
            r = self.session.get(url, timeout=8)
            data = r.json()
            quotes = data.get("quotes") or []
            if not quotes:
                return None

            # Prefer exact matches
            company_lower = company_name.lower()
            for qinfo in quotes:
                longname = qinfo.get("longname", "").lower()
                if longname and company_lower in longname:
                    return qinfo.get("symbol")

            # Return first symbol as fallback
            return quotes[0].get("symbol")
        except Exception:
            return None

    def fetch_news_for_company(self, company_name: str, ticker: Optional[str], max_results: int = 20) -> List[Dict]:
        all_items = []
        seen = set()

        # Company name
        for article in self.fetch_google_news(company_name, max_results):
            if article["title"] not in seen:
                seen.add(article["title"])
                all_items.append(article)

        # Ticker
        if ticker:
            for article in self.fetch_google_news(ticker, max_results):
                if article["title"] not in seen:
                    seen.add(article["title"])
                    all_items.append(article)

        # Company + stock
        for article in self.fetch_google_news(f"{company_name} stock", max_results // 2):
            if article["title"] not in seen:
                seen.add(article["title"])
                all_items.append(article)

        print(f"\nTotal unique articles: {len(all_items)}")
        return all_items[:max_results]

    # Scrape Article Text
    def scrape_article_content(self, url: str) -> Optional[str]:
        print("\nDEBUG: Starting scrape process")

        if not url:
            print("FAIL: No URL provided")
            return None

        # Newspaper4k first
        if NEWSPAPER_AVAILABLE:
            print("Trying newspaper4k")
            try:
                art = NewspaperArticle(url)
                art.download()
                art.parse()
                if art.text and len(art.text) > 200:
                    print(f"Success: newspaper4k extracted {len(art.text)} characters")
                    return art.text[:5000]
                else:
                    print(f"newspaper4k got text but too short ({len(art.text) if art.text else 0} chars)")
            except Exception as e:
                print(f"newspaper4k failed: {type(e).__name__}: {str(e)[:100]}")
        else:
            print("newspaper4k not available")

        # BeautifulSoup fallback
        print("Trying BeautifulSoup scraping")
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            print(f"HTTP request successful (status: {r.status_code}, {len(r.text)} bytes)")
        except Exception as e:
            print(f"FAIL: HTTP request failed: {type(e).__name__}: {str(e)[:100]}")
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove junk
        for tag in soup(["script", "style", "header", "footer", "aside", "iframe", "button", "form"]):
            tag.decompose()
        print("   Cleaned HTML")

        # Try <article>
        print("   Searching for <article> tag...")
        article = soup.find("article")
        if article:
            text = article.get_text(" ", strip=True)
            print(f"   Found <article> tag with {len(text)} characters")
            if len(text) > 200:
                print(f"   SUCCESS: Using <article> tag content")
                return text[:5000]
            else:
                print(f"   <article> tag too short, continuing...")
        else:
            print("   No <article> tag found")

        # Try content selectors
        selectors = [
            'div.article-body', 'div.entry-content', 'div.post-content',
            'main', 'section.article-body', 'div[itemprop="articleBody"]',
        ]
        print(f"Trying {len(selectors)} CSS selectors...")
        for sel in selectors:
            block = soup.select_one(sel)
            if block:
                text = block.get_text(" ", strip=True)
                print(f"Found '{sel}' with {len(text)} characters")
                if len(text) > 200:
                    print(f"SUCCESS: Using '{sel}' content")
                    return text[:5000]
                else:
                    print(f"'{sel}' too short, continuing...")
            else:
                print(f"Selector '{sel}' not found")

        # Paragraph aggregation
        print("Trying paragraph aggregation...")
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        print(f"Found {len(paras)} total <p> tags")
        paras = [p for p in paras if len(p) > 40]
        print(f"{len(paras)} paragraphs longer than 40 chars")
        if len(paras) >= 3:
            text = " ".join(paras)
            print(f"Combined paragraph text: {len(text)} characters")
            if len(text) > 200:
                print(f"SUCCESS: Using aggregated paragraphs")
                return text[:5000]
            else:
                print(f"Aggregated text too short")
        else:
            print(f"Not enough valid paragraphs found")

        print("FAIL: All extraction methods failed")
        return None


    # Chunked FinBERT Sentiment
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if not text or len(text.strip()) < 10:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "label": "neutral", "compound": 0.0}

        # Use chunking to avoid token limit issues with the FinBERT model
        max_tokens = 400  # Never set this to 512 because that's the max token input and increasing it might lead to some errors
        stride = 100  # Increased overlap between chunks for better context

        # First, encode the entire text to see how many tokens we have
        # Use truncation here too to avoid the warning
        all_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )

        # Safely get token count without triggering warnings
        token_count = len(all_ids)
        print(f"Total tokens in text: {token_count}")

        # If text is short enough, process it directly
        if token_count <= max_tokens:
            print(f"Text fits in single chunk, processing directly")
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                add_special_tokens=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits.cpu()

            probs = torch.nn.functional.softmax(logits[0], dim=-1).tolist()
        else:
            # Actually chunking the text
            print(f"Splitting into chunks (max {max_tokens} tokens each)...")

            chunks = []
            i = 0
            chunk_num = 0
            while i < len(all_ids):
                chunk_ids = all_ids[i:i + max_tokens]
                chunks.append(chunk_ids)
                chunk_num += 1
                print(f"Chunk {chunk_num}: {len(chunk_ids)} tokens")

                if i + max_tokens >= len(all_ids):
                    break
                i += max_tokens - stride

            print(f"Created {len(chunks)} chunks")

            # Process each chunk and accumulate logits
            total = None
            for idx, chunk_ids in enumerate(chunks, 1):
                # Decode the chunk(use skip_special_tokens to get clean text)
                chunk_txt = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)

                # Now create the actual model input with strict truncation
                # This ensures we never exceed 512 tokens
                inputs = self.tokenizer(
                    chunk_txt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                    add_special_tokens=True
                ).to(self.device)

                # Get the actual input size
                input_length = inputs['input_ids'].shape[1]

                with torch.no_grad():
                    logits = self.model(**inputs).logits.cpu()

                total = logits if total is None else total + logits
                print(f"Processed chunk {idx}/{len(chunks)} ({input_length} tokens)")

            # Average the logits across chunks
            if total is None:
                # If all chunks failed, return neutral as the base case
                print(f"All chunks failed, returning neutral sentiment")
                return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "label": "neutral", "compound": 0.0}

            avg = (total / len(chunks))[0]
            probs = torch.nn.functional.softmax(avg, dim=-1).tolist()

        # Map probs(labels)
        out = {self.id2label[i]: p for i, p in enumerate(probs)}
        for k in ["positive", "negative", "neutral"]:
            out.setdefault(k, 0.0)

        out["compound"] = out["positive"] - out["negative"]
        out["label"] = max(["positive", "negative", "neutral"], key=lambda x: out[x])

        return out

    # Weighted average sentiment
    def calculate_weighted_average(self, articles: List[Dict]) -> Dict:
        if not articles:
            return {"weighted_average": 0.0, "simple_average": 0.0}

        scores = []
        weighted_scores = []
        weights = []

        for a in articles:
            c = a["sentiment"]["compound"]
            scores.append(c)
            w = 1.0

            # Length weight
            L = a.get("text_length", 0)
            if L > 1500:
                w *= 1.4
            elif L > 600:
                w *= 1.2

            # Recency weight
            pub = a.get("published")
            if pub:
                try:
                    dt = datetime.fromisoformat(pub)
                except Exception:
                    dt = None
                if dt:
                    age = datetime.utcnow() - dt
                    if age < timedelta(hours=6):
                        w *= 1.5
                    elif age < timedelta(days=1):
                        w *= 1.3
                    elif age < timedelta(days=3):
                        w *= 1.1

            # Source credibility
            src = a.get("source", "").lower()
            if any(s in src for s in ["reuters", "bloomberg", "financial times"]):
                w *= 1.3
            elif any(s in src for s in ["cnbc", "marketwatch", "yahoo"]):
                w *= 1.15

            weights.append(w)
            weighted_scores.append(c * w)
            a["analysis_weight"] = round(w, 2)

        total_w = sum(weights)
        weighted_avg = sum(weighted_scores) / total_w if total_w else 0.0
        simple_avg = sum(scores) / len(scores) if scores else 0.0

        return {"weighted_average": weighted_avg, "simple_average": simple_avg, "weights": weights}

    # Full pipeline
    def analyze_company_news(self, company_name: str, max_articles: int = 20):
        print(f" ANALYZING NEWS FOR: {company_name}")

        ticker = self.lookup_ticker_top1(company_name)
        print(f"Ticker: {ticker}")

        articles = self.fetch_news_for_company(company_name, ticker, max_articles)

        if not articles:
            return {"error": f"No news found for {company_name}"}

        # Scraping loop
        analyzed = []
        for i, art in enumerate(articles, 1):
            print(f"\n[{i}/{len(articles)}] {art['title'][:80]}")
            print(f" URL: {art['url']}")
            url = art["url"]

            text = self.scrape_article_content(url)

            # DEBUG OUTPUT - Show what we're analyzing(sometimes it might be analysing a consent page or a cookies page or something like that)
            print("\n" + "─" * 70)
            print("Text being analyzed:")
            print("─" * 70)

            if text:
                art["full_text"] = text
                art["text_length"] = len(text)
                art["scraped"] = True
                print(f"Successfully scraped article ({len(text)} characters)")
                print(f"\nFirst 500 characters:\n{text[:500]}")
                if len(text) > 500:
                    print(f"\n... [+{len(text) - 500} more characters]")
            else:
                fallback = f"{art['title']} {art['summary']}"
                art["text_length"] = 0
                art["scraped"] = False
                text = fallback
                print(f"Could not scrape article - using fallback ({len(text)} characters)")
                print(f"\nFallback text (title + summary):\n{text}")

            print("─" * 70)

            sentiment = self.analyze_sentiment(text)
            art["sentiment"] = sentiment
            analyzed.append(art)

            comp = sentiment["compound"]
            print(f"\n Sentiment: {sentiment['label'].upper()} ({comp:+.3f})")
            print(
                f"   Positive: {sentiment['positive']:.3f} | Negative: {sentiment['negative']:.3f} | Neutral: {sentiment['neutral']:.3f}")

            time.sleep(0.35)

        stats = self.calculate_weighted_average(analyzed)
        W = stats["weighted_average"]

        overall = "POSITIVE" if W > 0.15 else "NEGATIVE" if W < -0.15 else "NEUTRAL"

        return {
            "company": company_name,
            "ticker": ticker,
            "weighted_sentiment": W,
            "simple_average": stats["simple_average"],
            "overall_sentiment": overall,
            "articles": analyzed,
            "timestamp": datetime.now().isoformat()
        }

    # Print everything out
    def print_summary(self, results):
        if "error" in results:
            print(results["error"])
            return

        print("SUMMARY")

        print(f"Company: {results['company']}")
        print(f"Ticker : {results.get('ticker')}")
        print(f"Overall sentiment: {results['overall_sentiment']}")
        print(f"Weighted score  : {results['weighted_sentiment']:+.3f}")
        print(f"Simple average  : {results['simple_average']:+.3f}")

        dist = {"positive": 0, "negative": 0, "neutral": 0}
        scraped_count = 0
        for a in results["articles"]:
            dist[a["sentiment"]["label"]] += 1
            if a.get("scraped"):
                scraped_count += 1

        print(f"\nArticles successfully scraped: {scraped_count}/{len(results['articles'])}")
        print("\nSentiment Distribution:")
        for k, v in dist.items():
            print(f"  {k.capitalize():8}: {v}")



# Main body
if __name__ == "__main__":
    analyzer = GoogleNewsSentimentAnalyzer()

    company = input("Company name: ").strip()
    if not company:
        print("Error: company name required.")
        exit(1)

    n = input("Max articles (default 20): ").strip()
    max_n = int(n) if n.isdigit() else 20

    res = analyzer.analyze_company_news(company, max_articles=max_n)
    analyzer.print_summary(res)

    # Save result
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{company.replace(' ', '_')}_sentiment_{ts}.json"

    to_save = json.loads(json.dumps(res))  # deep copy
    for a in to_save.get("articles", []):
        if "full_text" in a:
            a["text_preview"] = a["full_text"][:250] + "..."
            del a["full_text"]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2)

    print(f"\nSaved → {filename}")