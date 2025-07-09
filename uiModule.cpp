#include <ncurses.h>
#include <cstring>

// g++ -Wall -o uiModule.o uiModule.cpp -lncurses

WINDOW* create_newwin(int height, int width, int starty, int startx);
void write_win(WINDOW* win, char* str);

/* Initialize global variables */
WINDOW* trns_win;
int ch;
int trns_x;
int trns_y;

int main() {
    initscr(); /* initialize ncurses mode */ 
    cbreak(); /* disable line buffering */
    keypad(stdscr, TRUE); /* allow use of function keys */
    curs_set(0); /* make the cursor invisible */
    noecho(); /* prevent typed characters from echoing to the terminal */
    refresh();

    trns_win = create_newwin(LINES, int(COLS / 3), 0, 0);
    while((ch = getch()) != KEY_F(1)) {
        /* If the terminal is changing sizes change the window size */ 
        if (ch == KEY_RESIZE) { 
            delwin(trns_win);
            trns_win = create_newwin(LINES, int(COLS / 3), 0, 0);
        }
    }

    /* exit ncurses */
    endwin();
    return 0;
}

/* Helper functions */

void write_win(WINDOW* win, char* str) {
    /* Write one character at a time to the given window */
    int xmax;
    int ymax;
    int xbeg;
    int ybeg;
    int text_padding = 2;

    getbegyx(win, ybeg, xbeg);
    getmaxyx(win, ymax, xmax);

    int win_width = xmax - xbeg;
    int text_width = win_width - text_padding;
    int win_height = ymax - ybeg;
    int text_height = win_height - text_padding;

    /* Print one character at a time to the window and include wrapping */
    int cur_x = xbeg + int(text_padding / 2);
    int cur_y = ybeg + int(text_padding / 2);
    for (int i; i<strlen(str); i++) {
        /* print current character to the window */
        mvwaddch(win, cur_y, cur_x, str[i]);
        /* refresh the window after printing */
        wrefresh(win);
        
        /* iterate position values */
        cur_x++;
        if (cur_x > text_width) {
            cur_x = xbeg + int(text_padding / 2);
            cur_y++;
        }

        /* If we run out of space just stop printing */
        if (cur_y > win_height) {
            return;
        }
    }
}

WINDOW* create_newwin(int height, int width, int starty, int startx) {
    WINDOW* win;
    win = newwin(height, width, starty, startx);
    box(win, 0, 0); /* 0, 0 uses default characters for the vertical and horizontal lines */
    wrefresh(win);
    return win;
}