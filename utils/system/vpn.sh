# vpn
alias vpn1='export https_proxy=http://10.176.58.101:7890 http_proxy=http://10.176.58.101:7890 all_proxy=socks5://10.176.58.101:7891'
alias vpn2='export https_proxy=http://10.176.52.116:7890 http_proxy=http://10.176.52.116:7890 all_proxy=socks5://10.176.52.116:7891'
alias dvpn='unset https_proxy http_proxy all_proxy'

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY
env | grep -i proxy   # 此处最好没有任何 http/https/all_proxy