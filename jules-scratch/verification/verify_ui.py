from playwright.sync_api import sync_playwright, Page, expect

def test_new_ui(page: Page):
    """
    This test verifies that the new UI is rendered correctly.
    """
    try:
        print("Navigating to http://localhost:5174")
        page.goto("http://localhost:5174")
        page.wait_for_load_state("networkidle")
        print("Page loaded successfully")

        print("Checking for heading")
        expect(page.get_by_role("heading", name="PrivSyn")).to_be_visible()
        print("Heading found")

        print("Checking for lead text")
        expect(page.get_by_text("A Tool for Differentially Private Data Synthesis")).to_be_visible()
        print("Lead text found")

        print("Taking screenshot")
        page.screenshot(path="jules-scratch/verification/verification.png")
        print("Screenshot taken")

    except Exception as e:
        print(f"An error occurred: {e}")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    test_new_ui(page)
    browser.close()
