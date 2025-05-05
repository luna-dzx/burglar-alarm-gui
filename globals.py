import pygame
pygame.font.init()

font_46 = pygame.font.Font("OpenDyslexicNerdFont-Regular.otf", 46)
font_40 = pygame.font.Font("OpenDyslexicNerdFont-Regular.otf", 40)
font_36 = pygame.font.Font("OpenDyslexicNerdFont-Regular.otf", 36)
font_30 = pygame.font.Font("OpenDyslexicNerdFont-Regular.otf", 30)
font_22 = pygame.font.Font("OpenDyslexicNerdFont-Regular.otf", 22)
font_16 = pygame.font.Font("OpenDyslexicNerdFont-Regular.otf", 16)

fonts = {
    "small": font_22,
    "medium": font_30,
    "big": font_40
}

def select_font(size):
    if size == 0: # small:
        fonts["big"] = font_30
        fonts["medium"] = font_22
        fonts["small"] = font_16

    elif size == 1: # medium:
        fonts["big"] = font_40
        fonts["medium"] = font_30
        fonts["small"] = font_22

    else: # big:
        fonts["big"] = font_46
        fonts["medium"] = font_36
        fonts["small"] = font_30

select_font(1)
font_small = font_22
font_medium = font_30
font_big = font_40

TEXT_GAP_BIG = -10
TEXT_GAP_MEDIUM = -10

colours = {
    "background": (0,0,0),
    "foreground": (200,200,200),
    "button-outline": (150,150,150),
    "button-hover": (250,250,250),
    "button-click": (200,200,200),
    "text": (255,255,255),
    "error-text": (255,50,50),
    "success-text": (50,255,50)
}

symbols = {
    "lock-closed": "\uf023",
    "lock-open": "\uf2fc",
    "checked-box": "\U000f0c52",
    "empty-box": "\U000f0131",
    "camera": "\ueada",
    "dot": "\ueb8a",
    "error": "\uea87",
    "back": "\U000f030d",
    "home": "\ueb06",
    "check": "\uf00c",
    "cog": "\uf013"
}

SCROLL_WEIGHT = 0.1
