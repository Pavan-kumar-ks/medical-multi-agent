"""Scrapy settings for MedOrchestrator doctor scraper."""

BOT_NAME = "medorchestrator_scraper"
SPIDER_MODULES = ["spiders"]
NEWSPIDER_MODULE = "spiders"

# ── Politeness ────────────────────────────────────────────────────────────────
DOWNLOAD_DELAY         = 1.5          # seconds between requests
RANDOMIZE_DOWNLOAD_DELAY = True
CONCURRENT_REQUESTS    = 4
AUTOTHROTTLE_ENABLED   = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY   = 10.0

# ── Timeouts & limits ────────────────────────────────────────────────────────
DOWNLOAD_TIMEOUT       = 25
CLOSESPIDER_ITEMCOUNT  = 30           # stop after 30 doctors found
CLOSESPIDER_TIMEOUT    = 60           # hard stop after 60 s
DEPTH_LIMIT            = 3

# ── Playwright integration ────────────────────────────────────────────────────
DOWNLOAD_HANDLERS = {
    "http":  "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "args": ["--no-sandbox", "--disable-dev-shm-usage"],
}
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 20_000   # ms

# ── Middleware & pipelines ────────────────────────────────────────────────────
DOWNLOADER_MIDDLEWARES = {
    "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
    "middlewares.RotateUserAgentMiddleware": 400,
}

ITEM_PIPELINES = {
    "pipelines.CleanDoctorPipeline": 100,
    "pipelines.DeduplicatePipeline": 200,
}

# ── Misc ─────────────────────────────────────────────────────────────────────
ROBOTSTXT_OBEY        = False         # hospital sites often block bots via robots.txt
LOG_LEVEL             = "WARNING"
FEEDS                 = {}            # runner controls output path
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
