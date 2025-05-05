import pygame
from globals import *

class Button:
    def __init__(self, rect, label="", outline_width=2, font_size = 1):
        self.rect = pygame.Rect(rect[0],rect[1],rect[2],rect[3])
        self.outline_width = outline_width
        self.label = label
        self.font_size = font_size

    def set_pos(self, pos):
        self.rect = pygame.Rect(pos[0], pos[1], self.rect.width, self.rect.height)

    def set_rect(self, rect):
        self.rect = pygame.Rect(rect[0],rect[1],rect[2],rect[3])

    def process(self, event):
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                return True

        return False

    def render(self, surface):

        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]

        outline_colour = colours["button-outline"]
        hovering = self.rect.collidepoint(mouse_pos)
        if hovering:
            if mouse_pressed:
                outline_colour = colours["button-click"]
            else:
                outline_colour = colours["button-hover"]

        #pygame.draw.rect(surface, colour, self.rect)
        pygame.draw.rect(surface, outline_colour, self.rect, self.outline_width)

        if self.label != "":
            font = fonts["small"]
            if self.font_size == 1:
                font = fonts["medium"]
            elif self.font_size == 2:
                font = fonts["big"]
            text = font.render(self.label, True, outline_colour)
            surface.blit(text, (self.rect.centerx - text.get_rect().width/2, self.rect.centery - text.get_rect().height/2))

        return mouse_pressed and hovering


class UserScrollGrid:
    def __init__(self, rect, columns, padding, scroll_bar_width = 20, outline_width=2):
        self.rect = pygame.Rect(rect[0],rect[1],rect[2]-scroll_bar_width,rect[3])
        self.columns = columns
        self.padding = padding
        self.outline_width = outline_width
        self.scroll_bar_width = scroll_bar_width

        self.render_surface()
        self.max_scroll = self.inner_height - self.rect.height
        self.scroll = 0

        self.scroll_grabbed = False

    def set_pos(self, pos):
        self.rect.topleft = pos

    def render_surface(self):
        self.inner_height = 2000
        self.surface = pygame.Surface((self.rect.width,self.inner_height))
        self.surface.fill(colours["background"])
        self.buttons = []

        elem_width = self.rect.width/self.columns - self.padding
        elem_size = (elem_width, elem_width)
        gap = (self.rect.width - self.columns * elem_size[0]) / (self.columns + 1)

        users = ["Lucy", "Bob", "Jerma", "Geraldine", "Bartholemew", "Grindle"]

        for i,name in enumerate(users):
            xi = i%self.columns
            yi = i//self.columns
            
            x = gap * (xi+1) + elem_size[0] * xi
            y = self.padding * (yi+1) + elem_size[1] * yi

            bounding_rect = (x,y,elem_size[0],elem_size[1])

            pygame.draw.rect(self.surface, colours["foreground"], bounding_rect, self.outline_width)
            centre_x = x + elem_size[0] / 2
            y += 10
            name_text = fonts["medium"].render(name, True, colours["text"])
            rect = name_text.get_rect()
            text_x = centre_x - rect.width/2

            self.surface.blit(name_text, (text_x, y))
            pygame.draw.rect(self.surface, colours["foreground"], (text_x - 20, y, rect.width + 40, rect.height), self.outline_width)

            y = bounding_rect[0] + elem_size[1] - 80
            bw = elem_size[0] * 0.5
            button = Button((centre_x - bw*0.5, y, bw, 60), "Delete User")
            self.buttons.append(button)

            button.render(self.surface)



    def process(self, event):
        if event.type == pygame.MOUSEWHEEL:
            self.scroll -= event.y * SCROLL_WEIGHT * self.rect.height
            if self.scroll > self.max_scroll:
                self.scroll = self.max_scroll
            if self.scroll < 0:
                self.scroll = 0
        if event.type == pygame.MOUSEBUTTONUP:
            self.scroll_grabbed = False

    def render(self, surface):


        self.render_surface()

        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]

        scroll_rect = pygame.Rect(self.rect.right-1, self.rect.top, self.scroll_bar_width+1, self.rect.height)

        bar_height = (self.rect.height / self.inner_height) * self.rect.height
        scroll_offset = (self.scroll / self.rect.height) * bar_height

        if mouse_pressed:
            if scroll_rect.collidepoint(mouse_pos):
                self.scroll_grabbed = True

        if self.scroll_grabbed:
            y = mouse_pos[1] - bar_height / 2 - self.rect.top
            if y < 0:
                y = 0
            if y + bar_height > self.rect.height:
                y = self.rect.height - bar_height

            scroll_offset = y
            self.scroll = (scroll_offset / bar_height) * self.rect.height
                

        # cropped surface
        surface.blit(self.surface, self.rect.topleft, (0, self.scroll, self.rect.width, self.rect.height))

        # outlines
        pygame.draw.rect(surface, colours["foreground"], self.rect, self.outline_width)
        pygame.draw.rect(surface, colours["foreground"], scroll_rect, self.outline_width)
        # scroll rectangle
        pygame.draw.rect(surface, colours["foreground"], (self.rect.right,self.rect.top+scroll_offset,self.scroll_bar_width,bar_height))


class Slider:
    def __init__(self, rect, steps, snap=False, width=1, select_width=7, offset = 0):

        self.rect = pygame.Rect(rect[0],rect[1],rect[2],rect[3])
        self.start = (rect[0], rect[1] + rect[3]*0.5)
        self.end = (rect[0]+rect[2], self.start[1])
        self.height = rect[3]

        self.steps = steps
        self.snap = snap
        self.width = width
        self.advance = self.rect.width / ( len(steps) - 1 )

        self.select_offset = (float(offset) / (len(steps) - 1.0)) * self.rect.width
        self.select_width = select_width
        self.queue_snap = False

        self.just_clicked = False

    def set_rect(self,rect):
        self.rect = pygame.Rect(rect[0],rect[1],rect[2],rect[3])
        self.start = (rect[0], rect[1] + rect[3]*0.5)
        self.end = (rect[0]+rect[2], self.start[1])
        self.height = rect[3]
        self.advance = self.rect.width / ( len(self.steps) - 1 )


    def render(self, surface):

        self.labels = []
        for label in self.steps:
            self.labels.append(fonts["small"].render(label, True, colours["text"]))

        to_return = self.select_offset
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]

        if mouse_pressed and self.rect.collidepoint(mouse_pos):
            self.select_offset = mouse_pos[0] - self.rect.left
            self.queue_snap = True
            if not self.snap:
                to_return = self.select_offset
        else:
            if self.snap and self.queue_snap:
                for i in range(len(self.labels)):
                    if self.select_offset < (i+0.5)*self.advance:
                        self.select_offset = i*self.advance
                        to_return = i*self.advance
                        break
                self.queue_snap = False

        pygame.draw.line(surface, colours["foreground"], self.start, self.end, self.width)

        y_start = self.rect.top + 2
        y_end = self.rect.bottom - 2
        for i in range(len(self.steps)):
            x = self.rect.left + i*self.advance
            pygame.draw.line(surface, colours["foreground"], (x,y_start), (x,y_end), self.width)

            surface.blit(self.labels[i],(x - self.labels[i].get_rect().width // 2,y_end))

        pygame.draw.rect(surface, colours["foreground"],
                         (self.rect.left + self.select_offset - self.select_width/2, self.rect.top,
                          self.select_width, self.rect.height+1))

        if self.just_clicked and not mouse_pressed:
            self.just_clicked = mouse_pressed
            return (to_return / self.rect.width) * (len(self.steps) - 1)

        self.just_clicked = mouse_pressed
        return None


class TextField:
    def __init__(self, pos, centred=(True,False), max_length=-1,
                 numbers_only = False, always_active = False, outline_width=2,
                 min_width = 0, confirmable = True):

        self.active = always_active
        self.always_active = always_active
        self.text = ""
        self.numbers_only = numbers_only
        self.max_length = max_length
        self.pos = pos
        self.centred = centred
        self.outline_width = outline_width
        self.min_width = min_width
        self.height = fonts["medium"].render(" ", False, (0,0,0)).get_rect().height
        self.width = min_width
        self.confirmable = confirmable

    def set_pos(self, pos):
        self.pos = pos

    def process(self, event):

        if not self.active:
            return ""

        if event.type == pygame.MOUSEBUTTONUP:
            if not self.always_active:
                mouse_pos = pygame.mouse.get_pos()
                rect = pygame.Rect(0,0,self.width, self.height)

                x,y = self.get_raw_pos()

                if rect.collidepoint((mouse_pos[0] - x, mouse_pos[1] - y)):
                    self.active = True
                else:
                    self.active = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                if len(self.text) <= 1:
                    self.text = ""
                else:
                    self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                if self.confirmable:
                    text = self.text
                    self.text = ""
                    return text
            else:

                if self.max_length != -1 and len(self.text) >= self.max_length:
                    return ""

                if self.numbers_only:
                    if event.unicode.isdigit():
                        self.text += event.unicode
                else:
                    if event.unicode.isalpha() or event.unicode.isdigit() or event.unicode == " ":
                        self.text += event.unicode

        return ""

    def get_raw_pos(self):
        x = self.pos[0]
        y = self.pos[1]

        if self.centred[0]:
            x -= self.width/2
        if self.centred[1]:
            y -= self.height/2

        return x,y


    def render(self, surface):

        mouse_pressed = pygame.mouse.get_pressed()[0]
        mouse_pos = pygame.mouse.get_pos()

        string = self.text
        if len(string) == 0:
            string = " "
        text = fonts["medium"].render(" " + string + " ", True, colours["text"])
        
        self.height = text.get_rect().height
        width = max(text.get_rect().width, self.min_width)
        self.width = width

        x,y = self.get_raw_pos()

        surface.blit(text, (x + (self.width - text.get_rect().width) / 2,y))

        if mouse_pressed and not self.always_active:
            rect = text.get_rect()
            rect.width = width
            if rect.collidepoint((mouse_pos[0] - x, mouse_pos[1] - y)):
                self.active = True
            else:
                self.active = False


        outline_colour = colours["button-outline"]
        if self.active:
            outline_colour = colours["button-hover"]




        pygame.draw.rect(surface, outline_colour, (x,y,width,text.get_rect().height), self.outline_width)

class CheckBox:
    def __init__(self, pos, checked=False):
        self.checked = checked
        self.pos = pos

    def set_pos(self, pos):
        self.pos = pos

    def process(self, event):

        if event.type == pygame.MOUSEBUTTONUP:
            text = fonts["big"].render(symbols["empty-box"], False, (0,0,0))
            text_pos = (self.pos[0] - text.get_rect().width/2, self.pos[1] - text.get_rect().height/2)
            mouse_pos = pygame.mouse.get_pos()

            if text.get_rect().collidepoint((mouse_pos[0] - text_pos[0], mouse_pos[1] - text_pos[1])):
                self.checked = not self.checked
                return self.checked

        return None

    def get_width(self):

        text = fonts["big"].render(symbols["empty-box"], False, (0,0,0))
        return text.get_rect().width

    def render(self, surface):

        string = symbols["empty-box"]

        if self.checked:
            string = symbols["checked-box"]

        text = fonts["big"].render(string, True, colours["text"])
        text_pos = (self.pos[0] - text.get_rect().width/2, self.pos[1] - text.get_rect().height/2)

        surface.blit(text, text_pos)

