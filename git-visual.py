#!/usr/bin/env python3
# 2022 Sirio Balmelli

from datetime import date
from dateutil.relativedelta import relativedelta
from enum import Enum
from os import chdir, environ, getcwd, path
from subprocess import run

import matplotlib.pyplot as plt
import re


class Period(Enum):
    """
    Human readable time period, encoding in number of months.
    """
    Monthly = 1
    Quarterly = 3
    Yearly = 12

    def next(self, date):
        return date + relativedelta(months=self.value)

    def start(self, date):
        if self == Period.Monthly:
            return date.replace(day=1)
        elif self == Period.Quarterly:
            return date.replace(day=1, month=(1 + int((date.month + 2) / 3 - 1) * 3))  # noqa: E501
        elif self == Period.Yearly:
            return date.replace(day=1, month=1)

    def format(self, date):
        if self == Period.Monthly:
            return date.strftime("%Y-%m")
        elif self == Period.Quarterly:
            return f'Q{int((date.month + 2) / 3)} {date.strftime("%Y")}'
        elif self == Period.Yearly:
            return date.strftime("%Y")


# static regex matchers for gather()
re_names = ['insertions', 'deletions', 'files', 'commits']
re_total = re.compile(r'\s*total:')
re_author = re.compile(r'\s*(\S.*) (\<.*\>):')
re_stat = re.compile(rf'\s*({"|".join(re_names)}):\s+([0-9]+)\s+\(([0-9.]+)%\)')  # noqa: E501


def gather(start: date, end: date, period: Period, cwd: str) -> tuple:
    # TODO: ignore enormous commits (2022-07-08, Sirio Balmelli) #
    """
    Return a tuple of (periods, stats, label) where stats is:
    {user: {stat: [per-period values]}}

    @start : beginning of plot, will be rounded _down_ to beginning of 'period'
    @period : reporting interval
    """
    start = period.start(start)
    end = period.start(period.next(end))

    chdir(cwd)
    env = environ.copy()
    env["_GIT_LOG_OPTIONS"] = "--all"

    periods = []
    stats = {}
    author = None
    i = 0
    while start < end:
        periods.append(period.format(start))
        env["_GIT_SINCE"] = start.isoformat()
        # print(f'start: {start}')
        start = period.next(start)
        env["_GIT_UNTIL"] = start.isoformat()
        # print(f'next: {start}')

        proc = run(['git-quick-stats', '-T'],
                   capture_output=True,
                   env=env)

        for ln in proc.stdout.decode().splitlines():
            if re_total.match(ln):
                author = None
            elif m := re_author.match(ln):
                author = m.group(1)
                stats[author] = stats.get(author, {})
            elif m := re_stat.match(ln):
                if not author:
                    continue
                stat = m.group(1)

                st = stats[author].get(stat, [])
                st += [0 for j in range(len(st), i)]  # zero-pad leading
                st.append(int(m.group(2)))

                stats[author][stat] = st

        # zero-pad trailing entries
        i += 1  # use to zero-pad missing author entries
        for author in stats.keys():
            for stat in re_names:
                st = stats[author].get(stat, [])
                stats[author][stat] = st + [0 for j in range(len(st), i)]

    # sort stats dictionary: helps when debugging/printing/graphing
    stats = {auth: {stat: stats[auth][stat]
                    for stat in sorted(stats[auth].keys())}
             for auth in sorted(stats.keys())}

    return (periods, stats, path.basename(cwd))


def gather_merge(gathers: list) -> tuple:
    periods, stats, label = gathers[0]

    for i in range(1, len(gathers)):
        p2, s2, l2 = gathers[i]

        if not s2:
            continue
        # no metadata to merge different-length repos
        assert len(p2) == len(periods)

        def mauths():
            """
            Merge author names
            """
            return sorted(set(stats.keys()).union(s2.keys()))

        def mstats(auth):
            """
            Merge stat names for 'auth'
            """
            return sorted(set(stats.get(auth, {}).keys()).union(s2.get(auth, {}).keys()))  # noqa: E501

        def mval(auth, stat):
            """
            Merge values for '(auth, stat)'
            """
            t1 = stats.get(auth, {}).get(stat) or [0 for i in range(len(periods))]  # noqa: E501
            t2 = s2.get(auth, {}).get(stat) or [0 for i in range(len(periods))]
            return [t1[i] + t2[i] for i in range(len(periods))]

        stats = {auth: {stat: mval(auth, stat) for stat in mstats(auth)}
                 for auth in mauths()}

        label = f'{label}; {l2}'

    return periods, stats, label


# TODO: refactor plotting function, there is double compute here
# common variables for all plotting functions
plot_fmt = {'alpha': 0.7,
            'antialiased': True}
plot_save = None
# cargo cult programming at its finest
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def stackplot(periods, stats, label) -> None:
    authors = stats.keys()
    length = range(len(periods))

    y_num = {a: [stats[a]['insertions'][i] + stats[a]['deletions'][i]
                 for i in length]
             for a in authors}
    y_ins = {a: [stats[a]['insertions'][i] / (
                    sum(stats[k]['insertions'][i] for k in authors) or 1)
                 for i in length]
             for a in authors}
    y_del = {a: [stats[a]['deletions'][i] / (
                    sum(stats[k]['deletions'][i] for k in authors) or 1)
                 for i in length]
             for a in authors}

    # plot total changes by author
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # Reduce horizontal space between axes
    fig.subplots_adjust(hspace=0)

    ax[0].stackplot(periods, y_ins.values(), labels=y_ins.keys(), **plot_fmt)
    ax[0].set_ylabel('% Insertions')
    ax[0].yaxis.tick_right()
    ax[0].margins(y=0)

    ax[1].stackplot(periods, y_num.values(), labels=y_num.keys(), **plot_fmt)
    ax[1].set_ylabel('Lines of Code')
    ax[1].margins(y=0)

    ax[2].stackplot(periods, y_del.values(), labels=y_del.keys(), **plot_fmt)
    ax[2].set_ylabel('% Deletions')
    ax[2].yaxis.tick_right()
    ax[2].margins(y=0)

    ax[0].set_title(label)
    ax[1].legend(loc='best')

    # TODO: user a proper path library throughout this whole tool
    if plot_save:
        plt.draw()
        fig.savefig(f'{plot_save}/{periods[-1]} - {label} - stackplot.pdf',
                    bbox_inches='tight', format='pdf')
    else:
        plt.show()


def barchart(periods, stats, label) -> None:
    # TODO: graph total volume (2022-07-08, Sirio Balmelli) #

    authors = stats.keys()
    length = range(len(periods))

    fig, ax = plt.subplots()
    label_fmt = {'fontsize': 'x-small'}

    ax.set_xticks([n for n in length])
    ax.set_xticklabels(periods)

    # Bar placement is the _center_ of the bar, x is the _center_ of all bars:
    # when aligning remove 1 bar-width, corresponding to half
    # on each side of the group.
    width = 1.0 / (len(authors) + 1)  # leave 1 bar of space between groups
    offset = width * (len(authors) - 1) / 2
    i = 0
    for auth in authors:
        x = [n - offset + (width * i) for n in length]
        i += 1

        # insertions
        a_val = [stats[auth]["insertions"][i] for i in length]
        a_pct = [int(stats[auth]['insertions'][i] / (
                        sum(stats[k]['insertions'][i] for k in authors) or 1)
                     * 100)
                 for i in length]

        # deletions stacked above insertions, all red, only 1 legend for all
        one_lbl = 'deletions' if i == 1 else ''  # only label red column once
        b_val = [stats[auth]["deletions"][i] for i in length]
        # prevent dividing by 2 when one or both is zero
        b_pct = [int(stats[auth]['deletions'][i] / (
                        sum(stats[k]['deletions'][i] for k in authors) or 1)
                     * 100)
                 for i in length]
        b_pct = [int((a_pct[i] + b_pct[i]) / ((bool(a_pct[i]) + bool(b_pct[i])) or 2))  # noqa: E501
                 for i in length]
        b = ax.bar(x, b_val, width=width, label=one_lbl, color='red',
                   bottom=a_val, **plot_fmt)
        ax.bar_label(b, labels=[f'{p}%' for p in b_pct],
                     padding=4, label_type='edge', **label_fmt)

        # build 'a' _after_ 'b' so the deletion legend is shown first :P
        a = ax.bar(x, a_val, width=width, label=auth, **plot_fmt)
        ax.bar_label(a, labels=[f'{a_pct[i]}%' if (a_pct[i] and a_pct[i] != b_pct[i]) else ''  # noqa: E501
                                for i in length],
                     padding=0, label_type='center', **label_fmt)  # noqa: E501

    ax.set_title(label)
    ax.set_ylabel('Total Lines of Code')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              ncol=len(authors) + 1,  # +1 is the deletions legend
              fontsize='x-small')
    ax.margins(y=0.25)  # more space at the top helps legibility

    # TODO: user a proper path library throughout this whole tool
    if plot_save:
        plt.draw()
        fig.savefig(f'{plot_save}/{periods[-1]} - {label} - barchart.pdf',
                    bbox_inches='tight', format='pdf')
    else:
        plt.show()


# TODO: overlapping histograms of commit size by author

if __name__ == "__main__":
    import argparse

    # args
    desc = 'Git repo Visual Statistics Tool'
    ap = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument(
        'Period',
        nargs='?', choices=Period.__members__,
        default='Quarterly',
        help=('Reporting period: time represented by each graph tick')
        )
    ap.add_argument(
        'Start',
        nargs='?', type=date.fromisoformat,
        default=date.today().replace(month=1, day=1),
        help=('Start date in ISO (YYYY-mm-dd) format.'
              ' Actual start date will be rounded to the previous Period.')
        )

    ap.add_argument(
        'End',
        nargs='?', type=date.fromisoformat,
        default=date.today(),
        help=('End date in ISO (YYYY-mm-dd) format.')
        )

    ap.add_argument(
        '-p', '--paths', metavar='PATH',
        nargs='+', dest='paths', type=str,
        help=('Operate on repo at PATH instead of current dir.')
        )
    ap.add_argument(
        '-l', '--label', metavar='LABEL',
        dest='label', type=str,
        help=('Text label to put on graphs.')
        )
    ap.add_argument('-i', '--individual', dest='ind', action="store_true",
                    help=('Generate graph(s) for each individual repo,'
                          ' in addition to the total graph.')
                    )
    ap.add_argument('-w', '--write', metavar='PATH',
                    dest='write', type=str,
                    help=('Write graphs as PDFs to PATH')
                    )

    args = ap.parse_args()

    # TODO: clean up: use argparse defaults
    if args.paths:
        paths = [path.abspath(p) for p in args.paths]
    else:
        paths = [getcwd()]
    if args.write:
        plot_save = args.write

    # TODO: parallelism (2022-06-28, Sirio Balmelli) #
    each = [gather(args.Start, args.End, Period[args.Period], p)
            for p in paths]
    if args.ind:
        for periods, stats, label in each:
            # TODO: add a command-line switch (2022-06-28, Sirio Balmelli) #
            # stackplot(periods, stats, label)
            barchart(periods, stats, label)

    periods, stats, label = gather_merge(each)
    if args.label:
        label = args.label
    else:
        label = f'repo(s): {label}'
    # TODO: add a command-line switch (2022-06-28, Sirio Balmelli) #
    stackplot(periods, stats, label)
    barchart(periods, stats, label)
