#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <time.h>
using namespace std;
const int MAXN = 1e5 + 10;
const double fexp = 1e-6;
const int MERJIE = 1000;

bool matrixLAPP[MERJIE][MAXN];
int length;

struct node
{
    int x, y;
    double m;
};
node erjie[1440000];


bool cmp(node a, node b)
{
    if (fabs(a.m - b.m) < fexp && a.x == b.x) return a.y < b.y;
    if (fabs(a.m - b.m) < fexp) return a.x < b.x;
    return a.m > b.m;
}

bool judge(int X, int Y, int chosen, int j) 
    switch(chosen)
    {
        case 1: ///x^y
            return matrixLAPP[X][j] && matrixLAPP[Y][j];
            break;
        case 2: /// -(x^y)
            return !(matrixLAPP[X][j] && matrixLAPP[Y][j]);
            break;
        case 3:/// x u y
            return matrixLAPP[X][j] || matrixLAPP[Y][j];
            break;
        case 4:/// -(x u y)
            return !(matrixLAPP[X][j] || matrixLAPP[Y][j]);
            break;
        case 51:/// x ^ !y
            return matrixLAPP[X][j] && (!matrixLAPP[Y][j]);
            break;
        case 52:/// !x ^ y
            return (!matrixLAPP[X][j]) && matrixLAPP[Y][j];
            break;
        case 62:/// x u !y
            return matrixLAPP[X][j] || (!matrixLAPP[Y][j]);
            break;
        case 61:/// !x u y
            return (!matrixLAPP[X][j]) || matrixLAPP[Y][j];
            break;
        case 7:/// !(x <> y)
            return !(matrixLAPP[X][j] == matrixLAPP[Y][j]);
            break;
        case 8:/// x <> y
            return (matrixLAPP[X][j] == matrixLAPP[Y][j]);
            break;
    }
}

double H(int X)
{
    int cnt[2];
    memset(cnt, 0, sizeof(cnt));
    for (int j = 0; j < length; j++) {
        if (matrixLAPP[X][j] == 0) {
            cnt[0]++;
        }
        else if (matrixLAPP[X][j] == 1) {
            cnt[1]++;
        }
    }
    double p[2];
    for (int i = 0; i < 2; i++) {
        if (cnt[i] == 0) p[i] = 0;
        else p[i] = double(cnt[i]) / length;
    }

    double ans = 0.0;
    for (int i = 0; i < 2; i++) {
        if (fabs(p[i] - 0.0) < 1e-6) continue;
        ans += (p[i] * log(p[i]));
    }
    return -ans;
}

double H(int X, int Y, int chosen) 
{
    int cnt[2];
    memset(cnt, 0, sizeof(cnt));
    for (int j = 0; j < length; j++) {
        if (judge(X, Y, chosen, j) == 0) {
            cnt[0]++;
        }
        else if (judge(X, Y, chosen, j) == 1) {
            cnt[1]++;
        }
    }
    double p[2];
    memset(p, 0, sizeof(p));
    for (int i = 0; i < 2; i++) {
        if (cnt[i] == 0) p[i] = 0;
        else p[i] = double(cnt[i]) / length;
    }

    double ans = 0.0;
    for (int i = 0; i < 2; i++) {
        if (fabs(p[i] - 0.0) < 1e-8) continue;
        ans += (p[i] * log(p[i]));
    }
    return -ans;
}

double H(int X, int Y)
{
    int cnt[4];
    memset(cnt, 0, sizeof(cnt));
    for (int j = 0; j < length; j++) {
        if (!matrixLAPP[X][j] && !matrixLAPP[Y][j]) {
            cnt[0]++;
        }
        else if (!matrixLAPP[X][j] && matrixLAPP[Y][j]) {
            cnt[1]++;
        }
        else if (matrixLAPP[X][j] && !matrixLAPP[Y][j]) {
            cnt[2]++;
        }
        else if (matrixLAPP[X][j] && matrixLAPP[Y][j]) {
            cnt[3]++;
        }
    }

    double p[4];
    for (int i = 0; i < 4; i++) {
        if (cnt[i] == 0) p[i] = 0;
        else p[i] = cnt[i] * 1.0 / length;
    }

    double ans = 0;
    for (int i = 0; i < 4; i++) {
        if (fabs(p[i] - 0.0) < 1e-6 || p[i] < 0) continue;
        ans += (p[i] * log(p[i]));
    }
    return -ans;
}

double H(int Z, int X, int Y, int chosen)
{
    int cnt[4];
    memset(cnt, 0, sizeof(cnt));
    for (int j = 0; j < length; j++) {
        if (!matrixLAPP[Z][j] && !judge(X, Y, chosen, j)){
            cnt[0]++;
        }
        else if (!matrixLAPP[Z][j] && judge(X, Y, chosen, j)){
            cnt[1]++;
        }
        else if (matrixLAPP[Z][j] && !judge(X, Y, chosen, j)){
            cnt[2]++;
        }
        else if (matrixLAPP[Z][j] && judge(X, Y, chosen, j)){
            cnt[3]++;
        }
    }

    double p[4];
    for (int i = 0; i < 4; i++) {
        p[i] = cnt[i] * 1.0 / length;
    }


    double ans = 0;
    for (int i = 0; i < 4; i++) {
        if (fabs(p[i] - 0.0) < 1e-6 || p[i] < 0) continue;
        ans += (p[i] * log(p[i]));
        //printf("%.9lf ", p[i]);
    }
    //cout << endl;
    //cout << "ans:" << ans << endl;
    return -ans;
}

double U(int X, int Y) /// U(X | Y)
{
    double HEX = H(X);
    if (fabs(HEX) < fexp) return 1.0;
    //cout << H(X) << " " << H(Y) << " " << H(X, Y) << endl;
    return (HEX + H(Y) - H(X, Y)) / HEX;
}

double U(int Z, int X, int Y, int chosen) 
{
    double HEZ = H(Z);
    if (fabs(HEZ) < fexp) return 1.0;
    return (HEZ + H(X, Y, chosen) - H(Z, X, Y, chosen)) / HEZ;
}


bool isremove[1024];

int main()
{

    freopen("G:\\370v.txt", "r", stdin);
    printf("test\n");
    int n;
    scanf("%d", &n);
    char str[MAXN];
    memset(isremove, 0, sizeof(isremove));
    for (int i = 0; i < n; i++) {
        scanf("%s", str);
        //printf("%d line\n", i);
        length = strlen(str);
        int cntone = 0;
        for (int j = 0; j < length; j++) {
        	
            matrixLAPP[i][j] = (str[j] - '0');
            if (matrixLAPP[i][j] == 1) {
                cntone++;
            }
        }
        if (cntone <= 2) isremove[i] = 1;
    }
    printf("%d 完成数据读取\n", n);

	freopen("G:\\370w.txt", "w", stdout);
    for (int i = 0; i < n; i++) {
        if (isremove[i]) continue;
        for (int j = i + 1; j < n; j++) {
            if (isremove[i]) continue;
            for (int k = 0; k < n; k++) {
                if (k == i || k == j) continue;
                if (isremove[k]) continue;
                double res1 = U(k, i);
                double res2 = U(k, j);
                if (res1 < 0.4 && res2 < 0.4) {
                    int findex[15] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
                    int fcnt = 0;
                    double fvalue[15] = {0};
                    double resLapp = U(k, i, j, 1);
                    if (resLapp > 0.5) {
                        int cnt1 = 0, cnt2 = 0;
                        for (int loc = 0; loc < length; loc++) {
                            if (matrixLAPP[k][loc] != judge(i, j, 1, loc)){
                                cnt1++;
                            }
                            if (matrixLAPP[k][loc] != judge(i, j, 2, loc)){
                                cnt2++;
                            }
                        }
                        if (cnt1 == cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 1;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 2;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 1;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 2;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else if (cnt1 < cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 1;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 1;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 2;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 2;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                    }

                    resLapp = U(k, i, j, 3);
                    if (resLapp > 0.5) {
                        int cnt1 = 0, cnt2 = 0;
                        for (int loc = 0; loc < length; loc++) {
                            if (matrixLAPP[k][loc] != judge(i, j, 3, loc)){
                                cnt1++;
                            }
                            if (matrixLAPP[k][loc] != judge(i, j, 4, loc)){
                                cnt2++;
                            }
                        }
                        if (cnt1 == cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 3;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 4;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 3;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 4;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else if (cnt1 < cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 3;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 3;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 4;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 4;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                    }

                    resLapp = U(k, i, j, 51);
                    if (resLapp > 0.5) {
                        int cnt1 = 0, cnt2 = 0;
                        for (int loc = 0; loc < length; loc++) {
                            if (matrixLAPP[k][loc] != judge(i, j, 51, loc)){
                                cnt1++;
                            }
                            if (matrixLAPP[k][loc] != judge(i, j, 61, loc)){
                                cnt2++;
                            }
                        }
                        if (cnt1 == cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 51;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 61;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 51;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 61;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else if (cnt1 < cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 51;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 51;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 61;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 61;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                    }

                    resLapp = U(k, i, j, 52);
                    if (resLapp > 0.5) {
                        int cnt1 = 0, cnt2 = 0;
                        for (int loc = 0; loc < length; loc++) {
                            if (matrixLAPP[k][loc] != judge(i, j, 52, loc)){
                                cnt1++;
                            }
                            if (matrixLAPP[k][loc] != judge(i, j, 62, loc)){
                                cnt2++;
                            }
                        }
                        if (cnt1 == cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 52;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 62;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 52;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 62;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else if (cnt1 < cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 52;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 52;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 62;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 62;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                    }

                    resLapp = U(k, i, j, 7);
                    if (resLapp > 0.5) {
                        int cnt1 = 0, cnt2 = 0;
                        for (int loc = 0; loc < length; loc++) {
                            if (matrixLAPP[k][loc] != judge(i, j, 7, loc)){
                                cnt1++;
                            }
                            if (matrixLAPP[k][loc] != judge(i, j, 8, loc)){
                                cnt2++;
                            }
                        }
                        if (cnt1 == cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 7;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 8;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 7;
                                fvalue[fcnt] = resLapp;
                                fcnt++;
                                findex[fcnt] = 8;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else if (cnt1 < cnt2) {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 7;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 7;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                        else {
                            if (resLapp > fvalue[fcnt]) {
                                findex[fcnt] = 8;
                                fvalue[fcnt] = resLapp;
                            }
                            else if (fabs(resLapp - fvalue[fcnt]) < fexp) {
                                fcnt++;
                                findex[fcnt] = 8;
                                fvalue[fcnt] = resLapp;
                            }
                        }
                    }
                    for (int ind = 0; ind <= fcnt; ind++) {
                        if (findex[ind] == -1) continue;
                        printf("#%d#  V%04d V%04d V%04d  %.6lf %.6lf %.6lf\n", findex[ind], k+1, i+1, j+1, res1, res2, fvalue[ind]);
                    }
                }
            }
        }
    }


    return 0;
}
