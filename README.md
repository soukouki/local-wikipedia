# Local-Wikipedia

[日本語版はこちら](README-ja.md)

Local-Wikipedia is an MCP server that lets you bring Wikipedia offline—search and read articles after a quick one-time download.

Here’s why that matters:

1. Full-text search across Wikipedia
   - Most MCP servers only support exact title matches and basic redirect handling.
   - Because Local-Wikipedia stores the full text locally, you can run true full-text searches.
2. Keeps working even without the Internet after a one-time download
   - Since the data is saved locally, you can search and read Wikipedia articles even when you’re offline.
3. Handles high search frequency
   - No web API rate limits—iterate and refine your searches as much as you like.
   - Great for use cases that flexibly interpret queries and run repeated searches.
4. Easy to extend because the full text is local
   - With the entire corpus on hand, features that are hard on other MCP servers are much easier here.
     - Other MCP servers may be constrained by the web APIs they depend on; Local-Wikipedia isn’t.
   - For example, adding a “fetch a random article” feature is straightforward.

This MCP server is designed to pair nicely with small local LLMs. Even where large LLMs aren’t available, you can still search Wikipedia flexibly and retrieve information. We validated it with a compact, mobile-oriented model called Gemma 3n E4B, and it runs quickly even on CPU-only environments.

To make things small-LLM friendly, we made a few thoughtful choices:

1. Only two tools with minimal arguments, so LLMs can call them reliably.
2. Heuristic query correction to clean up over-specified or noisy inputs from an LLM.
3. Concise, situation-aware outputs so it runs fast even with short context windows and limited compute.
4. Advanced DB indexing for fast search and low memory use. Initial setup takes a bit longer, but once it’s done, it’s snappy.

## Features

This MCP server provides the following two tools.

### search_article

Flexibly searches for an article by the specified title and returns an appropriate result. Multiple search strategies are unified into a single tool to keep tool use simple for LLMs.

Search methods include:

1. Exact title match
2. Exact redirect match
3. Partial title match
4. Partial redirect match
5. Full-text search of the body

For 1 and 2, the article’s lead section (summary) is returned. For 3, 4, and 5, up to 20 results are returned.

There’s also a heuristic query-fix feature in case an LLM accidentally passes extra or irrelevant details to the tool.

### read_random_article

Fetches a random article from the specified language edition of Wikipedia.

## Setup

Everything is set up with Docker Compose. Follow the steps below.

Make sure Docker and Docker Compose are installed.

```bash
git clone https://github.com/soukouki/local-wikipedia.git
cd local-wikipedia
docker-compose up
```

On first run, the Wikipedia data for the language specified in config.yaml will be downloaded and indexed. This can take a while—roughly tens of minutes for Japanese and several hours for English. We recommend using a stable fiber connection for the download.

Once setup completes, the MCP server will start. By default, it listens on port 29423. Connect like this:

```json
{
  "servers": {
    "local-wikipedia": {
      "url": "http://local-wikipedia:29423"
    }
  }
}
```

## Technical Details

Local-Wikipedia uses the official Wikipedia dump data for pages and redirects, along with the Markdown-formatted full-text dataset published at [HuggingFaceFW/finewiki · Datasets at Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/finewiki#available-subsets).

For full-text search, it uses PGroonga to provide fast, memory-efficient search in both Japanese and English.